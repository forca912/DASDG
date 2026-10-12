"""Limited real-data functional pilot. No test-set evaluation or paper results."""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from compat import sha256
from inputs import causal_inputs, assert_input_invariance, select_starts, check_output_mapping
from models import build_model
from protocol import normalization_audit, atomic_checkpoint, write_json, write_csv, environment, digest
from vendor.train_v16b_external import build_correlation_adjacency, set_seed

ROOT = Path(__file__).resolve().parent
SEED = 20260808
CFG = {'input_steps': 12, 'output_steps': 12, 'hidden_dim': 64, 'embedding_dim': 16,
       'graph_dim': 32, 'top_k': 20, 'dropout': .1, 'graph_dropout': .1,
       'batch_size': 8, 'epochs': 3, 'training_windows': 512, 'validation_windows': 256,
       'learning_rate': .001, 'weight_decay': .0001, 'gradient_clip': 5.0}


class Windows(Dataset):
    def __init__(self, flow, mask, mean, std, maximum):
        self.inputs = causal_inputs(flow, mask, mean, std)
        self.targets = ((flow.astype(np.float32) - mean) / std).astype(np.float32)
        self.starts = select_starts(len(flow), maximum)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        start = int(self.starts[i])
        return torch.from_numpy(self.inputs[start:start + 12]), torch.from_numpy(self.targets[start + 12:start + 24])


def data(root):
    expected = json.loads((ROOT / 'expected_data.json').read_text(encoding='utf-8'))
    record = next(x for x in expected['legacy_protocol_matches'] if x['dataset'] == 'PEMS08' and x['seed'] == SEED)
    complete = root / 'processed/PEMS08/complete.npz'
    mask_path = root / f'masks_v2/PEMS08/persistent_rate_controlled/seed{SEED}_global0.1_nodes0.4_block12.npz'
    if sha256(complete) != record['complete_sha256'] or sha256(mask_path) != record['mask_sha256']:
        raise ValueError('Current PEMS08 data/mask differs from audited legacy identities')
    with np.load(complete, allow_pickle=False) as z:
        flow = z['flow'].astype(np.float32)
        split = z['split'].tolist()
        mean, std = float(z['mean'].flat[0]), float(z['std'].flat[0])
        features = z['features'] if 'features' in z else None
        file_metadata = {k: {'shape': list(z[k].shape), 'dtype': str(z[k].dtype)} for k in z.files}
    if flow.shape != (17856, 170) or split != [10713, 14284, 17856] or not np.isfinite(flow).all():
        raise ValueError('Data shape/split/finite values do not match published protocol')
    if mean != record['mean'] or std != record['std']:
        raise ValueError('Frozen mean/std differs from legacy run')
    normalization = normalization_audit(flow, split[0], mean, std, features)
    del features
    masks = {}
    with np.load(mask_path, allow_pickle=False) as z:
        for k, v in [('seed', SEED), ('block_length', 12), ('intended_global_missing_rate', .1),
                     ('affected_node_fraction', .4)]:
            if k not in z or not np.isclose(float(z[k]), v):
                raise ValueError('Mask metadata differs: ' + k)
        if 'observed_is_one' not in z or not bool(z['observed_is_one']):
            raise ValueError('Mask polarity invalid')
        start = 0
        for name, end in zip(('train', 'val', 'test'), split):
            m = z[name]
            if m.dtype != np.bool_ or m.shape != (end - start, flow.shape[1]):
                raise ValueError('Mask shape/dtype invalid: ' + name)
            masks[name] = m.copy()
            start = end
    evidence = {'dataset': 'PEMS08', 'seed': SEED, 'complete_sha256': sha256(complete),
                'mask_sha256': sha256(mask_path), 'legacy_identity': record['legacy_identity'],
                'normalization_audit': normalization, 'complete_npz_metadata': file_metadata,
                'normalization_values_changed': False, 'test_targets_used': False,
                'data_root': str(root), 'mean': mean, 'std': std}
    # Collect only candidate sidecar paths/sizes, never run pickle/model files.
    allowed_dirs = [root, root / 'processed/PEMS08', root / 'raw/PEMS08', root / 'PEMS08', root / 'raw']
    inventory = []
    for directory in allowed_dirs:
        if directory.is_dir():
            for p in sorted(directory.iterdir()):
                if p.is_file() and p.suffix.lower() in ('.csv', '.npy', '.npz', '.json', '.h5', '.pkl'):
                    inventory.append({'path': str(p), 'size_bytes': p.stat().st_size})
    evidence['sidecar_inventory_for_spatial_and_timestamp_contract'] = inventory[:100]
    return flow, masks, mean, std, split, evidence


def make_model(name, nodes, adjacency, device, compat_dir):
    if name == 'dasdg':
        return build_model('dasdg', nodes, adjacency, CFG).to(device)
    sys.path.insert(0, str(compat_dir))
    from pf_dgcrn.net import DGCRN
    a = torch.tensor(adjacency, dtype=torch.float32, device=device)
    # A.T is normalized independently, matching a two-direction transition contract.
    reverse = a.T / a.T.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return DGCRN(2, nodes, device, predefined_A=[a, reverse], seq_length=12, in_dim=3,
                  out_dim=12, rnn_size=64, node_dim=40, hyperGNN_dim=16, dropout=.1).to(device)


def forecast(model, name, x, y=None, batches_seen=0):
    if name == 'dasdg':
        return model(x)
    # Shared encoder values/mask/duration. Upstream's generic in_dim is 3 here.
    # Decoder auxiliaries remain unknown zero placeholders: never future masks
    # or durations taken from the dataset. This is an explicitly named adapter.
    features = x.permute(0, 3, 2, 1).contiguous()
    ycl = torch.zeros((len(x), 3, x.shape[2], 12), dtype=x.dtype, device=x.device)
    if model.training:
        if y is None:
            raise ValueError('Training teacher forcing requires training targets')
        ycl[:, 0] = y.permute(0, 2, 1)
    result = model(features, ycl=ycl, batches_seen=batches_seen, task_level=12)
    return result.squeeze(-1)


def node_time_probe(model, nodes, device):
    # Runs the real author's decoder and replaces only its final linear outputs
    # with distinguishable IDs, so the output reshape is checked independently.
    model.eval()
    records = []
    for batch in (2, 1):
        counter = [0]
        def hook(module, args, output):
            step = counter[0]
            counter[0] += 1
            ids = torch.arange(batch * nodes, device=device)
            return (10000 * (ids // nodes) + 1000 * step + ids % nodes).to(output.dtype).reshape(-1, 1)
        handle = model.fc_final.register_forward_hook(hook)
        try:
            with torch.no_grad():
                x = torch.zeros(batch, model.in_dim, nodes, 12, device=device)
                ycl = torch.zeros(batch, model.in_dim, nodes, 12, device=device)
                actual = model(x, ycl=ycl, batches_seen=0, task_level=12).cpu().numpy()
            if counter[0] != 12:
                raise ValueError('Unexpected number of decoder steps')
            records.append(check_output_mapping(actual, batch, nodes, 12))
        finally:
            handle.remove()
    return records


def future_target_probe(model, real_input):
    model.eval()
    encoded = real_input.permute(0, 3, 2, 1).contiguous()
    ycl = torch.zeros(len(real_input), 3, real_input.shape[2], 12, device=real_input.device)
    changed = ycl.clone()
    changed[:, 0] = 123456
    with torch.no_grad():
        set_seed(SEED)
        first = model(encoded, ycl=ycl, batches_seen=0, task_level=12)
        set_seed(SEED)
        second = model(encoded, ycl=changed, batches_seen=0, task_level=12)
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError('Nonfinite real-input future-target probe')
    if not torch.allclose(first, second, atol=1e-6, rtol=1e-6):
        raise ValueError('Adapted DGCRN eval depends on future true targets')
    return {'status': 'PASS', 'real_observed_input_used': True,
            'known_future_mask_or_duration_used': False,
            'max_abs_difference': float((first - second).abs().max().cpu())}


@torch.no_grad()
def evaluate(model, name, loader, device, std, mean):
    model.eval()
    sums = np.zeros((3, 12), dtype=np.float64)
    count = 0
    for i, (x, y) in enumerate(loader):
        # DGCRN upstream CUDA init is random even in eval; fix validation RNG.
        set_seed(SEED + 1000 + i)
        prediction = forecast(model, name, x.to(device)).cpu().to(torch.float64) * std + mean
        target = y.to(torch.float64) * std + mean
        error = prediction - target
        sums[0] += error.abs().sum((0, 2)).numpy()
        sums[1] += error.square().sum((0, 2)).numpy()
        sums[2] += target.abs().sum((0, 2)).numpy()
        count += len(y) * y.shape[2]
    rows = []
    for horizon in (3, 6, 9, 12, 'all'):
        s, c = (sums.sum(axis=1), count * 12) if horizon == 'all' else (sums[:, horizon - 1], count)
        rows.append({'horizon': str(horizon), 'diagnostic_mae': float(s[0] / c),
                     'diagnostic_rmse': float(np.sqrt(s[1] / c)),
                     'diagnostic_wape_percent': float(100 * s[0] / max(s[2], 1e-8)), 'pilot_only': True})
    if not all(np.isfinite(r[k]) for r in rows for k in ('diagnostic_mae', 'diagnostic_rmse', 'diagnostic_wape_percent')):
        raise ValueError('Nonfinite diagnostic metrics')
    return rows


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    args.output_created = True
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable, no silent CPU fallback')
    torch.set_num_threads(4)
    set_seed(SEED)
    flow, masks, mean, std, split, evidence = data(args.data_root.resolve())
    train_flow, val_flow = flow[:split[0]], flow[split[0]:split[1]]
    invariance = [assert_input_invariance(train_flow, masks['train'], mean, std, c)
                  for c in (12, split[0] // 2, split[0] - 12)]
    # Retain the previous normalization/fill/elapsed conventions exactly.
    from vendor.train_v16b_external import ForecastWindowDataset
    old = ForecastWindowDataset(train_flow, masks['train'], mean, std, 12, 12, 4)
    current = causal_inputs(train_flow, masks['train'], mean, std)
    if not np.array_equal(old.inputs, current):
        raise ValueError('New adapter does not match the old DASDG input convention')
    evidence['causal_invariance'] = invariance
    evidence['legacy_input_tensor_exact_match'] = True
    write_json(out / 'data_contract.json', evidence)
    datasets = {'train': Windows(train_flow, masks['train'], mean, std, CFG['training_windows']),
                'val': Windows(val_flow, masks['val'], mean, std, CFG['validation_windows'])}
    generator = torch.Generator().manual_seed(SEED)
    loaders = {n: DataLoader(d, batch_size=CFG['batch_size'], shuffle=(n == 'train'),
                generator=generator if n == 'train' else None, num_workers=0)
               for n, d in datasets.items()}
    write_json(out / 'window_indices.json', {n: d.starts.tolist() for n, d in datasets.items()})
    adjacency = build_correlation_adjacency((train_flow - mean) / std, 20, masks['train'])
    evidence['adjacency'] = {'kind': 'masked_training_correlation_proxy_not_geographic',
        'sha256_array_bytes': __import__('hashlib').sha256(adjacency.tobytes()).hexdigest()}
    model = make_model(args.model, flow.shape[1], adjacency, device, args.compat_dir)
    adapter = 'published_DASDG_forecast_path' if args.model == 'dasdg' else 'DGCRN_MD_author_code_zero_future_aux_corr_pilot'
    spec = {'model': args.model, 'adapter': adapter, 'pilot_only': True, 'fixture_only': True,
            'formal_result': False, 'configuration': CFG, 'data': evidence,
            'environment': environment(device), 'package_file_hashes': json.loads((ROOT / 'package_manifest.json').read_text()),
            'future_calendar_feature': 'disabled_constant_zero_not_fake_time_of_day',
            'selection_split': 'limited_validation_windows', 'test_evaluation_performed': False,
            'encoder_observation_information_equal': True,
            'encoder_features': ['causal_filled_value', 'observation_mask', 'elapsed_missing_duration'],
            'decoder_mask_duration': 'unknown_zero_placeholders_never_true_future_states',
            'normalization_calibration_boundary': 'complete_training_statistics_legacy_before_synthetic_masking',
            'cross_model_performance_comparison_permitted': False,
            'completed_job_skip': 'no; each coordinator invocation uses a new directory',
            'interrupted_training_resume': 'not supported in this functional pilot'}
    identity = digest(spec)
    write_json(out / 'run_config.json', {'identity': identity, **spec})
    if args.model == 'dgcrn':
        write_json(out / 'node_horizon_mapping.json', {'status': 'PASS',
            'method': 'identifiable marker hook at actual author fc_final',
            'cases': node_time_probe(model, flow.shape[1], device)})
        real_sample = torch.stack([datasets['train'][i][0] for i in (0, 1)]).to(device)
        write_json(out / 'eval_target_invariance.json', future_target_probe(model, real_sample))
    set_seed(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['learning_rate'], weight_decay=CFG['weight_decay'])
    rows, best, batches_seen = [], float('inf'), 0
    torch.cuda.reset_peak_memory_stats(device) if device.type == 'cuda' else None
    for epoch in range(1, CFG['epochs'] + 1):
        start = time.perf_counter()
        model.train()
        total, count = 0., 0
        for step, (x, y) in enumerate(loaders['train'], 1):
            optimizer.zero_grad(set_to_none=True)
            x, y = x.to(device), y.to(device)
            pred = forecast(model, args.model, x, y, batches_seen)
            if pred.shape != y.shape:
                raise ValueError('Training target/output identity shape mismatch')
            loss = torch.nn.functional.l1_loss(pred, y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            batches_seen += 1
            total += float(loss.detach()) * len(x)
            count += len(x)
            progress = {'status': 'running', 'model': args.model, 'epoch': epoch, 'epochs': CFG['epochs'],
                        'batch': step, 'batches': len(loaders['train']), 'pilot_only': True}
            write_json(out / 'progress.json', progress)
            if step == 1 or step % 8 == 0:
                print(f'PILOT model={args.model} epoch={epoch}/3 batch={step}/{len(loaders["train"])} loss={loss.item():.5f}', flush=True)
        metrics = evaluate(model, args.model, loaders['val'], device, std, mean)
        score = metrics[-1]['diagnostic_mae']
        if score < best:
            best = score
            atomic_checkpoint(out / 'best.pt', model.state_dict())
        rows.append({'epoch': epoch, 'train_l1_normalized': total / count,
                     'limited_validation_mae': score, 'seconds': time.perf_counter() - start,
                     'pilot_only': True})
        write_csv(out / 'history.csv', rows)
        print(f'PILOT_EPOCH_DONE model={args.model} epoch={epoch}/3 seconds={rows[-1]["seconds"]:.1f}', flush=True)
    model.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=True))
    metrics = evaluate(model, args.model, loaders['val'], device, std, mean)
    write_csv(out / 'limited_validation_diagnostics.csv', metrics)
    write_json(out / 'resource_diagnostics.json', {'pilot_only': True,
        'registered_parameters': sum(p.numel() for p in model.parameters()),
        'maximum_training_or_validation_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        'epoch_seconds': [r['seconds'] for r in rows],
        'timing_scope': 'limited_train_plus_limited_validation_plus_progress_writes_not_full_epoch_estimate'})
    write_json(out / 'progress.json', {'status': 'PILOT_COMPLETE', 'epochs': 3, 'model': args.model,
                                      'pilot_only': True, 'formal_training_completed': False})
    artifacts = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out / 'PILOT_DONE.json', {'identity': identity, 'status': 'PILOT_COMPLETE',
                                      'pilot_only': True, 'files': artifacts})
    print('PILOT_COMPLETE', args.model, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['dasdg', 'dgcrn'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--compat-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    try:
        run(args)
    except (Exception, SystemExit) as exc:
        import traceback
        if getattr(args, 'output_created', False):
            write_json(args.out / 'PILOT_FAILED.json', {'status': 'FAIL', 'pilot_only': True,
                       'model': args.model, 'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
