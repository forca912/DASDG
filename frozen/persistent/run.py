"""Preflight, sequential experiment queue, progress, and provenance-aware result audit."""
from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import torch
from protocol import (ROOT, all_jobs, digest, environment, job_path, load_data, lock, read_json,
                      read_csv, sha256, sources, verify_complete, write_csv, write_json)


def audit(out, config, stage, expected_sources=None, fixture=False):
    with lock(Path(out) / '.audit.lock'):
        return _audit(out, config, stage, expected_sources, fixture)


def _audit(out, config, stage, expected_sources=None, fixture=False):
    summary, horizons, statuses = [], [], []
    code = sources() if expected_sources is None else expected_sources
    for dataset, seed, name in all_jobs(config, stage):
        directory = job_path(out, dataset, seed, name)
        done = verify_complete(directory)
        if not done:
            status = 'NOT_STARTED'
            detail = {}
            if (directory / 'progress.json').exists():
                detail = read_json(directory / 'progress.json')
                status = 'PARTIAL'
            elif (directory / 'run_config.json').exists():
                status = 'PARTIAL'
            statuses.append({'dataset': dataset, 'seed': seed, 'model': name, 'status': status, **detail})
            continue
        spec = read_json(directory / 'run_config.json')
        if spec['sources'] != code or spec['protocol'] != config or spec['fixture_only'] != fixture:
            raise RuntimeError(f'Code/protocol/fixture mismatch: {directory}')
        if (spec['dataset'], spec['seed'], spec['model']) != (dataset, seed, name):
            raise RuntimeError('Condition key mismatch')
        recomputed = digest({k:v for k,v in spec.items() if k != 'identity'})
        if recomputed != done['identity']:
            raise RuntimeError('Run identity digest is inconsistent')
        efficiency = read_json(directory / 'efficiency.json')
        rows = read_csv(directory / 'test_metrics.csv')
        overall = next(r for r in rows if r['horizon'] == 'all')
        common = {'dataset': dataset, 'seed': seed, 'model': name, 'fixture_only': fixture}
        summary.append({**common, 'test_mae': float(overall['mae']), 'test_rmse': float(overall['rmse']),
                        'test_wape': float(overall['wape']), 'best_val_mae': efficiency['best_val_mae'],
                        'best_epoch': efficiency['best_epoch'], 'completed_epochs': efficiency['completed_epochs'],
                        'training_seconds': efficiency['training_seconds'],
                        'registered_parameters': efficiency['registered_parameters'],
                        'forecast_gradient_active_parameters': efficiency['forecast_gradient_active_parameters'],
                        'latency_per_sample_ms': efficiency['latency_per_sample_ms'],
                        'data_sha256': spec['data']['complete_sha256'], 'mask_sha256': spec['data']['mask_sha256'],
                        'run_dir': str(directory)})
        horizons.extend({**common, **row} for row in rows)
        statuses.append({**common, 'status': 'VERIFIED_COMPLETE'})
    for dataset in config['datasets']:
        for seed in config['seeds']:
            subset = [r for r in summary if r['dataset'] == dataset and r['seed'] == seed]
            if len({(r['data_sha256'],r['mask_sha256']) for r in subset}) > 1:
                raise RuntimeError('Models were evaluated using different data/masks')
    expected = len(all_jobs(config, stage))
    report = {'stage': stage, 'completed': len(summary), 'expected': expected,
              'status': 'PASS' if len(summary) == expected else 'INCOMPLETE',
              'fixture_only': fixture, 'runs': statuses}
    write_json(Path(out) / f'{stage}_audit.json', report)
    if summary:
        write_csv(Path(out) / f'{stage}_summary.csv', summary)
        write_csv(Path(out) / f'{stage}_horizon_metrics.csv', horizons)
    print(f'{stage}: {report["status"]} {len(summary)}/{expected} verified', flush=True)
    for r in statuses:
        if r['status'] not in ('NOT_STARTED', 'VERIFIED_COMPLETE'):
            print(f"  {r['dataset']} seed={r['seed']} {r['model']}: {r['status']} epoch={r.get('epoch','?')}", flush=True)
    return report


def discover(config):
    candidates = []
    base = Path('/root/autodl-tmp')
    for pattern in ('data', '*/data', '*/*/data', '*/*/*/data'):
        for path in base.glob(pattern):
            if all((path / 'processed' / d / 'complete.npz').exists() for d in config['datasets']):
                resolved = path.resolve()
                if resolved not in candidates:
                    candidates.append(resolved)
    valid = []
    for root in candidates:
        try:
            evidence = [load_data(root, d, s, config)[-1] for s in config['seeds'] for d in config['datasets']]
            valid.append((root, digest(evidence)))
        except (ValueError, KeyError, FileNotFoundError) as exc:
            print(f'DATA_CANDIDATE_REJECTED {root}: {exc}', flush=True)
    if not valid:
        raise FileNotFoundError('No valid frozen data root found. Locate data/processed/PEMS04/complete.npz and pass --data-root to its data directory.')
    if len({v[1] for v in valid}) > 1:
        raise RuntimeError('Multiple different valid data roots found; explicitly select --data-root: ' + ', '.join(str(v[0]) for v in valid))
    print('SELECTED_DATA_ROOT', valid[0][0], flush=True)
    return valid[0][0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['preflight', 'run', 'status', 'audit', 'bundle'])
    p.add_argument('--config', type=Path, default=ROOT / 'configs/primary.json')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/r1_robustness')
    p.add_argument('--stage', choices=['robustness', 'fairness', 'ablation', 'all'], default='robustness')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--fixture', action='store_true')
    args = p.parse_args()
    config = read_json(args.config)
    out = args.output.resolve()
    if args.fixture and 'SMOKE' not in str(out):
        raise ValueError('Fixture output must contain SMOKE')
    if args.action in ('status', 'audit'):
        with lock(out / '.queue.lock') if args.action == 'audit' else lock(out / '.status.lock'):
            report = audit(out, config, args.stage, fixture=args.fixture)
        if args.action == 'audit' and report['status'] != 'PASS':
            sys.exit(2)
        return
    if args.action == 'bundle':
        import tarfile
        with lock(out / '.queue.lock'):
            report = audit(out, config, args.stage, fixture=args.fixture)
            if report['status'] != 'PASS':
                raise RuntimeError('Cannot bundle an incomplete stage')
            archive = out.parent / f'{out.name}_{args.stage}_results_{time.strftime("%Y%m%d_%H%M%S")}.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                for path in sorted(out.rglob('*')):
                    if path.is_file() and path.suffix not in ('.pt', '.tmp', '.lock'):
                        tar.add(path, arcname=str(path.relative_to(out)))
            print('RESULT_BUNDLE', archive, flush=True)
            print('Checkpoints stay on server; this compact analysis bundle is not a checkpoint archive.', flush=True)
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available in this Python. Activate the training environment; do not reinstall PyTorch blindly.')
    torch.set_num_threads(config['training']['cpu_threads'])
    env = environment(torch.device(args.device))
    root = args.data_root.resolve() if args.data_root else discover(config)
    records = {}
    for seed in config['seeds']:
        for dataset in config['datasets']:
            records[f'{dataset}/{seed}'] = load_data(root, dataset, seed, config, args.fixture)[-1]
            method = records[f'{dataset}/{seed}']['normalization_audit']['matched_algorithm']
            print(f'NORMALIZATION_PASS dataset={dataset} seed={seed} algorithm={method} values_changed=False', flush=True)
    plan = {'data_root': str(root), 'config': config, 'sources': sources(), 'inputs': records,
            'environment': env, 'fixture_only': args.fixture}
    with lock(out / '.queue.lock'):
        pp = out / 'preflight.json'
        if pp.exists() and read_json(pp) != plan:
            raise RuntimeError('Existing output has a different preflight. Preserve it and select a new --output directory.')
        write_json(pp, plan)
        print(f'PREFLIGHT_PASS datasets={len(config["datasets"])} masks={len(records)} device={args.device}', flush=True)
        if args.action == 'preflight':
            return
        jobs = all_jobs(config, args.stage)
        for index, (dataset, seed, name) in enumerate(jobs, 1):
            directory = job_path(out, dataset, seed, name)
            print(f'[{index}/{len(jobs)}] {dataset} seed={seed} model={name}', flush=True)
            cmd = [sys.executable, '-u', str(ROOT / 'train.py'), '--config', str(args.config.resolve()),
                   '--data-root', str(root), '--dataset', dataset, '--seed', str(seed), '--model', name,
                   '--output', str(directory), '--device', args.device]
            if args.fixture:
                cmd.append('--fixture')
            log = out / 'logs' / f'{dataset}_{seed}_{name}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a', encoding='utf-8') as f:
                f.write(f'\nQUEUE_INVOCATION {time.strftime("%Y-%m-%d %H:%M:%S")}\n'); f.flush()
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding='utf-8', errors='replace', bufsize=1,
                                           env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
                for line in process.stdout:
                    print(line, end='', flush=True); f.write(line); f.flush()
                code = process.wait()
            if code:
                raise RuntimeError(f'Run failed (exit={code}); queue stopped. Inspect {log}; no automatic retry.')
        report = audit(out, config, args.stage, fixture=args.fixture)
        if report['status'] != 'PASS':
            raise RuntimeError('Queue ended with incomplete evidence')
        print('STAGE_COMPLETE', args.stage, flush=True)


if __name__ == '__main__':
    main()
