"""One fair-protocol run; resume from last completed epoch and audit all artifacts."""
from __future__ import annotations
import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from models import build_model, input_channels
from protocol import (atomic_checkpoint, digest, environment, load_data, lock, read_json,
                      sha256, sources, verify_complete, write_csv, write_json)
from vendor.train_v16b_external import (ForecastWindowDataset, build_correlation_adjacency,
                                     measure_latency, set_seed, metric_rows)


@torch.no_grad()
def evaluate_stream(model, loader, device, std, mean):
    """All-window, all-node metrics in raw units; no full prediction array in RAM."""
    model.eval()
    absolute = squared = denom = None
    count = 0
    for x, y in loader:
        prediction = model(x.to(device)).cpu().to(torch.float64) * std + mean
        target = y.to(torch.float64) * std + mean
        error = prediction - target
        a, b, c = error.abs().sum((0, 2)), error.square().sum((0, 2)), target.abs().sum((0, 2))
        if absolute is None:
            absolute, squared, denom = a, b, c
        else:
            absolute += a; squared += b; denom += c
        count += len(y) * y.shape[2]
    rows = []
    for horizon in (3, 6, 9, 12, 'all'):
        if horizon == 'all':
            a, b, c, n = absolute.sum(), squared.sum(), denom.sum(), count * len(absolute)
        else:
            a, b, c, n = absolute[horizon-1], squared[horizon-1], denom[horizon-1], count
        rows.append({'horizon': str(horizon), 'mae': float(a/n), 'rmse': float((b/n).sqrt()),
                     'wape': float(a / c.clamp_min(1e-8) * 100)})
    if not all(np.isfinite(r[k]) for r in rows for k in ('mae', 'rmse', 'wape')):
        raise RuntimeError('Non-finite evaluation metrics')
    return rows


def rng_state(generator):
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'loader': generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); generator.set_state(state['loader'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def train(args):
    config = read_json(args.config)
    t = config['training']
    if t['output_steps'] != 12 or t['epochs'] <= 0 or t['patience'] <= 0:
        raise ValueError('Invalid common training protocol')
    if args.fixture and 'SMOKE' not in str(args.output):
        raise ValueError('Fixture output path must contain SMOKE')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing silent CPU fallback')
    torch.set_num_threads(t['cpu_threads'])
    set_seed(args.seed)
    env = environment(device)
    flow, masks, mean, std, split, evidence = load_data(args.data_root, args.dataset, args.seed, config, args.fixture)
    if args.dataset not in config['datasets'] or args.seed not in config['seeds'] or args.model not in sum(config['stages'].values(), []):
        raise ValueError('Job outside registered protocol')
    spec = {'dataset': args.dataset, 'seed': args.seed, 'model': args.model, 'protocol': config,
            'data': evidence, 'sources': sources(), 'environment': env,
            'input_channels': input_channels(args.model),
            'selection_split': 'validation', 'evaluation_split': 'independent_test',
            'normalization': 'complete_training_flow_global_mean_std_legacy_protocol',
            'elapsed_initialization': 'legacy_zero_at_split_first_step',
            'fixture_only': args.fixture}
    identity = digest(spec)
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with lock(directory / '.run.lock'):
        if verify_complete(directory, identity):
            print('SKIP_VERIFIED_COMPLETE', directory, flush=True)
            return
        config_path = directory / 'run_config.json'
        if config_path.exists():
            if read_json(config_path)['identity'] != identity:
                raise RuntimeError('Existing run has different code/data/config/environment; choose a new output directory')
        else:
            if any(p.name != '.run.lock' for p in directory.iterdir()):
                raise RuntimeError('Unidentified files in output directory; refusing overwrite')
            write_json(config_path, {'identity': identity, **spec})
        segments = ((0, split[0]), (split[0], split[1]), (split[1], split[2]))
        datasets = {name: ForecastWindowDataset(flow[a:b], masks[name], mean, std,
                      t['input_steps'], t['output_steps'], t['max_delta_scale'])
                    for name, (a, b) in zip(('train', 'val', 'test'), segments)}
        generator = torch.Generator().manual_seed(args.seed)
        loaders = {name: DataLoader(data, batch_size=t['batch_size'], shuffle=name == 'train',
                    generator=generator if name == 'train' else None, num_workers=t['num_workers'],
                    pin_memory=device.type == 'cuda') for name, data in datasets.items()}
        adjacency = build_correlation_adjacency((flow[:split[0]]-mean)/std, t['top_k'], masks['train'])
        model = build_model(args.model, flow.shape[1], adjacency, t).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=t['learning_rate'], weight_decay=t['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=t['lr_factor'],
                     patience=t['lr_patience'], min_lr=t['min_learning_rate'])
        history, best, stale, best_epoch, seconds, start_epoch = [], float('inf'), 0, 0, 0.0, 1
        last_path = directory / 'last.pt'
        if last_path.exists():
            # This file is created by this local runner. Never load arbitrary downloaded .pt files.
            saved = torch.load(last_path, map_location='cpu', weights_only=False)
            if saved['identity'] != identity:
                raise RuntimeError('Checkpoint identity mismatch')
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
            history, best, stale, best_epoch, seconds = (saved[k] for k in ('history', 'best', 'stale', 'best_epoch', 'seconds'))
            start_epoch = saved['epoch'] + 1
            # last.pt is committed after best.pt. Reject a missing/altered best file.
            if sha256(directory / 'best.pt') != saved['best_sha256']:
                # Crash could occur between committing an improved best and last.
                previous = directory / 'best_previous.pt'
                if not previous.exists() or sha256(previous) != saved['best_sha256']:
                    raise RuntimeError('No matching best checkpoint for epoch resume')
                import shutil
                shutil.copyfile(previous, directory / 'best.pt')
            restore_rng(saved['rng'], generator)
            print(f'RESUME next_epoch={start_epoch} best_epoch={best_epoch}', flush=True)
        print(f'START model={args.model} dataset={args.dataset} seed={args.seed} device={device} train_windows={len(datasets["train"])}', flush=True)
        for epoch in range(start_epoch, t['epochs']+1):
            if stale >= t['patience']:
                break
            started = time.perf_counter()
            model.train(); total_loss = 0.0; count = 0
            for x, y in loaders['train']:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.l1_loss(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), t['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                total_loss += loss.item() * len(x); count += len(x)
            val = evaluate_stream(model, loaders['val'], device, std, mean)[-1]['mae']
            scheduler.step(val)
            elapsed = time.perf_counter() - started; seconds += elapsed
            history.append({'epoch': epoch, 'train_loss_norm': total_loss/count, 'val_mae_raw': val,
                            'learning_rate': optimizer.param_groups[0]['lr'], 'seconds': elapsed})
            if val < best - t['improvement_threshold']:
                best, stale, best_epoch = val, 0, epoch
                if (directory / 'best.pt').exists():
                    import shutil
                    shutil.copyfile(directory / 'best.pt', directory / 'best_previous.pt')
                atomic_checkpoint(directory / 'best.pt', model.state_dict())
            else:
                stale += 1
            atomic_checkpoint(last_path, {'identity': identity, 'epoch': epoch, 'model': model.state_dict(),
                       'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                       'history': history, 'best': best, 'stale': stale, 'best_epoch': best_epoch,
                       'seconds': seconds, 'rng': rng_state(generator), 'best_sha256': sha256(directory / 'best.pt')})
            write_csv(directory / 'history.csv', history)
            write_json(directory / 'progress.json', {'epoch': epoch, 'maximum_epochs': t['epochs'],
                       'best_epoch': best_epoch, 'best_val_mae': best, 'stale_epochs': stale,
                       'training_seconds': seconds, 'status': 'training', 'fixture_only': args.fixture})
            print(f'epoch={epoch:03d}/{t["epochs"]} val_MAE={val:.6f} best={best:.6f} stale={stale}/{t["patience"]} seconds={elapsed:.1f}', flush=True)
            if args.stop_after_epoch and epoch >= args.stop_after_epoch:
                print('TEST_CONTROL_STOP_AT_EPOCH', epoch, flush=True)
                return
        write_csv(directory / 'history.csv', history)
        model.load_state_dict(torch.load(directory / 'best.pt', map_location=device, weights_only=True))
        for name in ('val', 'test'):
            rows = evaluate_stream(model, loaders[name], device, std, mean)
            write_csv(directory / ('validation_metrics.csv' if name == 'val' else 'test_metrics.csv'), rows)
        sample = datasets['test'][0][0][None].to(device)
        # cuDNN-backed recurrent layers reject backward() in eval mode.  Use
        # training mode only for this parameter-activity probe, then restore
        # eval mode before the deterministic inference-latency measurement.
        model.train(); model.zero_grad(set_to_none=True)
        model(sample).sum().backward()
        active = sum(p.numel() for p in model.parameters() if p.grad is not None)
        model.zero_grad(set_to_none=True)
        model.eval()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        efficiency = measure_latency(model, sample, device, t['latency_warmup'], t['latency_repetitions'])
        efficiency.update({'registered_parameters': sum(p.numel() for p in model.parameters()),
                           'forecast_gradient_active_parameters': active, 'training_seconds': seconds,
                           'best_epoch': best_epoch, 'completed_epochs': len(history), 'best_val_mae': best,
                           'latency_protocol': 'forecast_only_batch1_perf_counter_device_sync_no_transfer',
                           'checkpoint_size_bytes': (directory / 'best.pt').stat().st_size,
                           'peak_inference_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None})
        write_json(directory / 'efficiency.json', efficiency)
        required = ('run_config.json', 'history.csv', 'validation_metrics.csv', 'test_metrics.csv', 'efficiency.json', 'best.pt')
        write_json(directory / 'DONE.json', {'identity': identity, 'fixture_only': args.fixture,
                   'files': {name: sha256(directory / name) for name in required}})
        verify_complete(directory, identity)
        write_json(directory / 'progress.json', {'status': 'complete', 'epoch': len(history), 'best_epoch': best_epoch,
                   'training_seconds': seconds, 'fixture_only': args.fixture})
        print('RUN_COMPLETE', directory, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--fixture', action='store_true', help='Synthetic software checks only; not manuscript evidence')
    p.add_argument('--stop-after-epoch', type=int, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.stop_after_epoch and not a.fixture:
        p.error('stop-after-epoch is a fixture test hook only')
    train(a)


if __name__ == '__main__':
    main()
