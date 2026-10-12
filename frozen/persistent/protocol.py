"""Provenance, fail-closed input checks, atomic output, and OS-managed locks."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import os
import platform
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
HORIZONS = ('3', '6', '9', '12', 'all')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refusing empty metrics table')
    s = io.StringIO(newline='')
    writer = csv.DictWriter(s, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, s.getvalue())


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def atomic_checkpoint(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@contextmanager
def lock(path):
    """Locks release automatically after process exit/reboot; never delete lock files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open('a+b')
    f.seek(0, 2)
    if f.tell() == 0:
        f.write(b'0'); f.flush()
    f.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        f.close()
        raise RuntimeError(f'Another process is using {path.parent}; do not start duplicates') from exc
    try:
        yield
    finally:
        f.close()


def sources():
    files = list(ROOT.glob('*.py')) + list((ROOT / 'vendor').glob('*.py'))
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in sorted(files)}


def environment(device):
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
            'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
            'platform': platform.platform(), 'threads': torch.get_num_threads()}


def mask_path(root, dataset, seed, config):
    mask_root = config.get('mask_root', 'masks_v2')
    pattern = config.get('mask_pattern', 'persistent_rate_controlled')
    return Path(root) / mask_root / dataset / pattern / (
        f"seed{seed}_global{config['global_missing_rate']:g}_nodes"
        f"{config['affected_node_fraction']:g}_block{config['block_length']}.npz")


def mask_statistics(mask):
    lengths = []
    for node in range(mask.shape[1]):
        missing = np.r_[False, ~mask[:, node], False].astype(np.int8)
        changes = np.diff(missing)
        lengths.extend((np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)).tolist())
    runs = np.array(lengths, dtype=int)
    values, counts = np.unique(runs, return_counts=True)
    return {'realized_missing_rate': float((~mask).mean()),
            'realized_affected_node_fraction': float((~mask).any(axis=0).mean()),
            'run_count': len(lengths), 'run_length_mean': float(runs.mean()) if len(runs) else 0.0,
            'run_length_max': int(runs.max()) if len(runs) else 0,
            'run_length_histogram': {str(v): int(c) for v, c in zip(values, counts)}}


def normalization_audit(flow, train_end, mean, std, features=None):
    """Verify frozen statistics against their generating algorithm, never rewrite them.

    prepare_data.py originally reduces a multi-channel float32 matrix along
    axis 0. NumPy's accumulation order differs from a float64 flow reduction.
    Verify that original operation explicitly instead of relaxing tolerances.
    """
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError('Invalid normalization')
    train = flow[:train_end]
    precise = [float(train.mean(dtype=np.float64)),
               float(max(train.std(dtype=np.float64), 1e-6))]
    saved = np.array([mean, std], dtype=np.float64)
    candidates = {}
    if features is not None:
        if features.ndim != 3 or features.shape[:2] != flow.shape or features.shape[2] < 1:
            raise ValueError('Invalid features shape for legacy normalization verification')
        if not np.isfinite(features).all():
            raise ValueError('Non-finite features during normalization verification')
        if not np.array_equal(flow, features[..., 0]):
            raise ValueError('Flow does not match features[..., 0]; refusing normalization fallback')
        # Preserve the exact layout and dtype operations in released prepare_data.py.
        old = features[:train_end].astype(np.float32).reshape(-1, features.shape[-1])
        candidates['legacy_features_float32_axis0'] = [
            float(old.mean(axis=0)[0]), float(np.maximum(old.std(axis=0), 1e-6)[0])]
    candidates['flow_float64_training_only'] = precise
    matched = [name for name, pair in candidates.items()
               if np.allclose(saved, pair, rtol=1e-6, atol=1e-6)]
    if not matched:
        raise ValueError('Normalization verification failed: ' + json.dumps({
            'saved_mean_std': saved.tolist(), 'recomputed_candidates': candidates,
            'rtol': 1e-6, 'atol': 1e-6}))
    return {'verification': 'PASS', 'matched_algorithm': matched[0],
            'all_matching_algorithms': matched, 'saved_mean_std': saved.tolist(),
            'recomputed_candidates': candidates, 'rtol': 1e-6, 'atol': 1e-6,
            'features_flow_channel_equal': features is not None,
            'relative_difference_from_float64': (np.abs(saved-precise) /
                np.maximum(np.abs(precise), 1e-12)).tolist(),
            'normalization_values_changed': False}


def load_data(root, dataset, seed, config, fixture=False):
    complete_path = Path(root) / 'processed' / dataset / 'complete.npz'
    mp = mask_path(root, dataset, seed, config)
    with np.load(complete_path, allow_pickle=False) as z:
        flow = z['flow'].astype(np.float32)
        split = z['split'].tolist()
        mean, std = float(z['mean'].flat[0]), float(z['std'].flat[0])
    if flow.ndim != 2 or not np.isfinite(flow).all() or len(split) != 3:
        raise ValueError('Invalid flow or split array')
    if any(not isinstance(v, int) for v in split) or not 0 < split[0] < split[1] < split[2] == len(flow):
        raise ValueError('Invalid chronological split boundaries')
    if not fixture:
        expected = {'PEMS03': (26208, 358), 'PEMS04': (16992, 307), 'PEMS08': (17856, 170)}
        if flow.shape != expected[dataset]:
            raise ValueError(f'{dataset}: expected {expected[dataset]}, found {flow.shape}; verify dataset provenance')
        if split != [int(len(flow)*.6), int(len(flow)*.8), len(flow)]:
            raise ValueError('Split differs from the published chronological 60/20/20 protocol')
    with np.load(complete_path, allow_pickle=False) as z:
        features = z['features'] if 'features' in z else None
        normalization = normalization_audit(flow, split[0], mean, std, features)
    del features
    masks = {}
    stats = {}
    with np.load(mp, allow_pickle=False) as z:
        for key, expected in [('seed', seed), ('block_length', config['block_length']),
                              ('intended_global_missing_rate', config['global_missing_rate']),
                              ('affected_node_fraction', config['affected_node_fraction'])]:
            if key not in z or not np.isclose(float(z[key]), expected):
                raise ValueError(f'{mp}: missing or inconsistent metadata {key}')
        if config.get('mask_root') == 'masks_r1_robustness':
            expected_pattern = config.get('mask_pattern', 'regular')
            if 'mask_pattern' not in z or str(z['mask_pattern'].item()) != expected_pattern:
                raise ValueError(f'{mp}: missing or inconsistent metadata mask_pattern')
            if 'mask_generator_version' not in z:
                raise ValueError(f'{mp}: missing mask_generator_version')
        if 'observed_is_one' not in z or not bool(z['observed_is_one']):
            raise ValueError('Mask must declare observed_is_one=True')
        start = 0
        for name, end in zip(('train', 'val', 'test'), split):
            m = z[name]
            if m.dtype != np.bool_ or m.shape != (end-start, flow.shape[1]):
                raise ValueError(f'Invalid {name} mask shape/dtype')
            if end-start < config['training']['input_steps'] + config['training']['output_steps']:
                raise ValueError('Split too short for forecasting windows')
            masks[name] = m.copy()
            stats[name] = mask_statistics(m)
            if not fixture and abs(stats[name]['realized_missing_rate']-config['global_missing_rate']) > .005:
                raise ValueError(f'{name}: realized missing rate outside tolerance')
            if not fixture and abs(stats[name]['realized_affected_node_fraction']-config['affected_node_fraction']) > 1/flow.shape[1]:
                raise ValueError(f'{name}: affected-node fraction mismatch')
            start = end
    evidence = {'complete_sha256': sha256(complete_path), 'mask_sha256': sha256(mp),
                'shape': list(flow.shape), 'split': split, 'mean': mean, 'std': std,
                'mask_path': str(mp), 'mask_pattern': config.get('mask_pattern', 'persistent_rate_controlled'),
                'normalization_audit': normalization,
                'mask_statistics': stats, 'fixture_only': fixture}
    return flow, masks, mean, std, split, evidence


def all_jobs(config, stage='all'):
    names = sum(config['stages'].values(), []) if stage == 'all' else config['stages'][stage]
    return [(dataset, seed, name) for seed in config['seeds'] for dataset in config['datasets'] for name in names]


def job_path(out, dataset, seed, name):
    return Path(out) / 'runs' / dataset / f'seed{seed}' / name


def verify_complete(directory, expected_identity=None):
    directory = Path(directory)
    marker = directory / 'DONE.json'
    if not marker.exists():
        return None
    done = read_json(marker)
    if expected_identity is not None and done['identity'] != expected_identity:
        raise RuntimeError(f'Run identity mismatch: {directory}')
    required = {'run_config.json', 'history.csv', 'validation_metrics.csv', 'test_metrics.csv', 'efficiency.json', 'best.pt'}
    if set(done['files']) != required:
        raise RuntimeError(f'Incomplete completion manifest: {directory}')
    for name, expected in done['files'].items():
        if not (directory / name).is_file() or sha256(directory / name) != expected:
            raise RuntimeError(f'Output hash mismatch: {directory / name}')
    if read_json(directory / 'run_config.json')['identity'] != done['identity']:
        raise RuntimeError('Config identity mismatch')
    for split in ('validation', 'test'):
        rows = read_csv(directory / f'{split}_metrics.csv')
        if len(rows) != len(HORIZONS) or {r['horizon'] for r in rows} != set(HORIZONS):
            raise RuntimeError('Duplicate/missing horizon metrics')
        if not all(np.isfinite(float(r[k])) and float(r[k]) >= 0 for r in rows for k in ('mae', 'rmse', 'wape')):
            raise RuntimeError('Invalid metrics')
    return done
