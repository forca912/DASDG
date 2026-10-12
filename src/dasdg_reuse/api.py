"""Validated complete-label data import, training, resume and causal inference.

This API uses the preserved R1 model operators. It is a software reuse example,
not a replacement for the benchmark protocol or support for missing target labels.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ._engine.models import build_model
from ._engine.vendor.train_v16b_external import (
    ForecastWindowDataset, build_correlation_adjacency, set_seed,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    root = Path(__file__).parent
    return {str(p.relative_to(root)).replace('\\', '/'): sha256(p)
            for p in sorted(root.rglob('*.py'))}


def validate_history(values, observed):
    values, observed = np.asarray(values), np.asarray(observed)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError('values must be a real numeric [time, node] array')
    if observed.dtype != np.bool_ or observed.shape != values.shape:
        raise ValueError('observed must be boolean, same shape as values; True means observed')
    if not np.isfinite(values[observed]).all():
        raise ValueError('Observed values must be finite')
    if values.shape[1] < 1:
        raise ValueError('At least one node is required')
    # Unobserved values are never read by causal filling. Accept NaN there.
    converted = values.astype(np.float32)
    if not np.isfinite(converted[observed]).all():
        raise ValueError('Observed values exceed the float32 range')
    return converted, observed


def load_dataset(path):
    with np.load(path, allow_pickle=False) as z:
        missing = {'flow', 'observed', 'split'} - set(z.files)
        if missing:
            raise ValueError(f'Missing dataset keys: {sorted(missing)}')
        flow, observed, split = z['flow'], z['observed'], z['split']
    flow, observed = validate_history(flow, observed)
    if not np.isfinite(flow).all():
        raise ValueError('Training/evaluation require complete finite target labels; missing labels are unsupported')
    if split.shape != (3,) or not np.issubdtype(split.dtype, np.integer):
        raise ValueError('split must contain three integer chronological boundaries')
    a, b, c = map(int, split)
    if not 0 < a < b < c == len(flow):
        raise ValueError('Require 0 < train_end < val_end < total == len(flow)')
    return flow, observed, (a, b, c)


@dataclass(frozen=True)
class Config:
    model: str = 'dasdg'
    input_steps: int = 12
    output_steps: int = 12
    hidden_dim: int = 64
    embedding_dim: int = 16
    graph_dim: int = 32
    top_k: int = 20
    dropout: float = 0.1
    graph_dropout: float = 0.1
    max_delta_scale: float = 4.0
    batch_size: int = 32
    epochs: int = 150
    patience: int = 25
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    improvement_threshold: float = 0.00001
    seed: int = 20260808
    cpu_threads: int = 2

    def validate(self):
        if self.model not in ('dasdg', 'gwn_duration', 'gru_duration'):
            raise ValueError('Supported models: dasdg, gwn_duration, gru_duration')
        for key in ('input_steps', 'output_steps', 'hidden_dim', 'embedding_dim', 'graph_dim',
                    'top_k', 'batch_size', 'epochs', 'patience', 'cpu_threads'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('dropout', 'graph_dropout'):
            if not 0 <= getattr(self, key) < 1:
                raise ValueError(f'{key} must be in [0, 1)')
        for key in ('learning_rate', 'max_delta_scale'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        for key in ('weight_decay', 'improvement_threshold'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f'{key} must be finite and nonnegative')


def _device(name):
    device = torch.device(name)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only cpu and cuda devices are supported')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    return device


def _save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _rng(generator):
    numpy = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]),
            'torch': torch.get_rng_state(), 'loader': generator.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state, generator):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'])
    generator.set_state(state['loader'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


@torch.no_grad()
def _mae(model, loader, device, std):
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        error = (model(x.to(device)).cpu().double() - y.double()).abs() * std
        total += error.sum().item()
        count += error.numel()
    value = total / count
    if not np.isfinite(value):
        raise RuntimeError('Non-finite validation MAE')
    return value


@contextmanager
def _run_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+b')
    stream.seek(0, 2)
    if stream.tell() == 0:
        stream.write(b'0'); stream.flush()
    stream.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise RuntimeError('Another process is using this output directory') from exc
    try:
        yield
    finally:
        stream.close()


def fit(dataset, output, config=None, device='cpu', resume=False, stop_after_epoch=None):
    """Train or resume through an OS-managed exclusive output-directory lock."""
    with _run_lock(Path(output) / '.run.lock'):
        return _fit(dataset, output, config, device, resume, stop_after_epoch)


def _fit(dataset, output, config=None, device='cpu', resume=False, stop_after_epoch=None):
    """Train from complete labels; normalization uses training labels only.

    Checkpoints include optimizer, scheduler and RNG. Resume is at epoch boundaries.
    stop_after_epoch is an acceptance-test control and is excluded from run identity.
    """
    cfg = config or Config()
    cfg.validate()
    flow, observed, split = load_dataset(dataset)
    if min(split[0], split[1]-split[0], split[2]-split[1]) < cfg.input_steps + cfg.output_steps:
        raise ValueError('Each split must have enough steps for a forecasting window')
    dev = _device(device)
    torch.set_num_threads(cfg.cpu_threads)
    set_seed(cfg.seed)
    mean = float(flow[:split[0]].mean(dtype=np.float64))
    std = float(max(flow[:split[0]].std(dtype=np.float64), 1e-6))
    segments = [(0, split[0]), (split[0], split[1]), (split[1], split[2])]
    windows = [ForecastWindowDataset(flow[a:b], observed[a:b], mean, std,
               cfg.input_steps, cfg.output_steps, cfg.max_delta_scale) for a, b in segments]
    generator = torch.Generator().manual_seed(cfg.seed)
    loaders = [DataLoader(w, batch_size=cfg.batch_size, shuffle=i == 0,
               generator=generator if i == 0 else None, num_workers=0)
               for i, w in enumerate(windows)]
    adjacency = build_correlation_adjacency((flow[:split[0]]-mean)/std, cfg.top_k, observed[:split[0]])
    model = build_model(cfg.model, flow.shape[1], adjacency, asdict(cfg)).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, min_lr=1e-5)
    metadata = {'config': asdict(cfg), 'nodes': flow.shape[1], 'mean': mean, 'std': std,
                'dataset_sha256': sha256(dataset), 'sources': source_identity(),
                'torch': str(torch.__version__), 'numpy': str(np.__version__),
                'python': platform.python_version(), 'device': str(dev),
                'scope': 'custom_complete_label_reuse_example_not_benchmark',
                'elapsed_initialization': 'legacy_zero_at_split_first_step',
                'normalization': 'complete_training_labels_float64_only'}
    identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    last = out / 'last.pt'
    if resume and not last.is_file():
        raise FileNotFoundError(f'Resume checkpoint not found: {last}')
    if not resume and any(p.name != '.run.lock' for p in out.iterdir()):
        raise FileExistsError('Output directory is nonempty; use a new directory or explicit resume')
    epoch, best, stale, history, best_weights = 0, float('inf'), 0, [], None
    if resume:
        saved = torch.load(last, map_location='cpu', weights_only=True)
        if saved['identity'] != identity:
            raise ValueError('Resume rejected: data/config/code/environment identity mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        epoch, best, stale, history, best_weights = (saved[k] for k in ('epoch','best','stale','history','best_weights'))
        _restore_rng(saved['rng'], generator)
    for epoch in range(epoch + 1, cfg.epochs + 1):
        if stale >= cfg.patience:
            break
        model.train()
        for x, y in loaders[0]:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(model(x.to(dev)), y.to(dev))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
        val = _mae(model, loaders[1], dev, std)
        scheduler.step(val)
        if val < best - cfg.improvement_threshold:
            best, stale = val, 0
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        history.append({'epoch': epoch, 'validation_mae': val})
        _save(last, {'identity': identity, 'metadata': metadata, 'epoch': epoch,
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'rng': _rng(generator),
                    'best': best, 'stale': stale, 'history': history, 'best_weights': best_weights})
        print(f'epoch={epoch}/{cfg.epochs} validation_MAE={val:.6f}', flush=True)
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            return {'status': 'paused_at_epoch_boundary', 'epoch': epoch, 'identity': identity}
    model.load_state_dict(best_weights)
    test_mae = _mae(model, loaders[2], dev, std)
    _save(out / 'best.pt', {'metadata': metadata, 'model': best_weights})
    result = {'status': 'complete', 'identity': identity, 'epochs_completed': len(history),
              'best_validation_mae': best, 'test_mae': test_mae,
              'fixture_or_custom_data': True, 'benchmark_result': False,
              'history': history, 'metadata': metadata}
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


@torch.no_grad()
def predict(checkpoint, values, observed, device='cpu'):
    """Predict [horizon, node] in original units, reading history only.

    Supply the complete available historical prefix to retain carry-forward state.
    Values at unobserved positions may be NaN and do not influence predictions.
    """
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    meta = saved['metadata']
    cfg = Config(**meta['config'])
    cfg.validate()
    values, observed = validate_history(values, observed)
    if values.shape[1] != meta['nodes'] or len(values) < cfg.input_steps:
        raise ValueError('History node count mismatch or insufficient input steps')
    dev = _device(device)
    model = build_model(cfg.model, meta['nodes'], np.zeros((meta['nodes'], meta['nodes']), np.float32), asdict(cfg)).to(dev)
    model.load_state_dict(saved['model'])
    model.eval()
    # Dataset construction needs a dummy future; it is never supplied to the model.
    padded = np.concatenate([values, np.zeros((cfg.output_steps, meta['nodes']), np.float32)])
    mask = np.concatenate([observed, np.ones((cfg.output_steps, meta['nodes']), bool)])
    windows = ForecastWindowDataset(padded, mask, meta['mean'], meta['std'], cfg.input_steps, cfg.output_steps, cfg.max_delta_scale)
    x = windows[len(values)-cfg.input_steps][0][None].to(dev)
    result = model(x).cpu().numpy()[0] * meta['std'] + meta['mean']
    if not np.isfinite(result).all():
        raise RuntimeError('Non-finite inference output')
    return result
