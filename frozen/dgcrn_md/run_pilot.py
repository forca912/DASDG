"""Sequential, bounded real-data pilots; no auto retry, no formal results."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from compat import verify_sources, build_compat, sha256

ROOT = Path(__file__).resolve().parent


def dump(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def verify_package():
    manifest = json.loads((ROOT / 'package_manifest.json').read_text(encoding='utf-8'))
    for rel, expected in manifest.items():
        p = ROOT / rel
        if not p.resolve().is_relative_to(ROOT.resolve()) or sha256(p) != expected:
            raise ValueError('Package file changed: ' + rel)
    verify_sources(ROOT)
    return len(manifest)


def run_suite():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/traffic_missing_forecasting/data'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--timeout-per-model', type=int, default=1800)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if not 60 <= args.timeout_per_model <= 7200:
        parser.error('timeout-per-model must be 60..7200 seconds')
    count = verify_package()
    print(f'PACKAGE_VERIFIED files={count}; AUTHOR_SOURCE_VERIFIED', flush=True)
    # NumPy tests do not require GPU; perform them before any training process.
    subprocess.run([sys.executable, '-B', str(ROOT / 'tests_contract.py')], check=True)
    if args.validate_only:
        print('PASS_LOCAL_CONTRACT_TESTS_MODEL_EXECUTION_NOT_RUN', flush=True)
        return 0
    import numpy
    import pandas
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no automatic CPU fallback')
    if not args.data_root.is_dir():
        raise FileNotFoundError('DATA_ROOT_NOT_FOUND: ' + str(args.data_root))
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = args.out.resolve() if args.out else ROOT / ('pilot_results_' + stamp)
    if out == ROOT or ROOT / 'upstream' == out or ROOT / 'upstream' in out.parents:
        parser.error('Output must be a new directory outside upstream')
    out.mkdir(parents=True, exist_ok=False)
    status = {'status': 'RUNNING', 'pilot_only': True, 'formal_training_completed': False,
              'models': {}, 'planned_jobs': 2, 'output': str(out),
              'hard_timeout_seconds_per_model': args.timeout_per_model,
              'test_evaluation_performed': False,
              'gcnm_realdata_pilot': 'PENDING_SPATIAL_HISTORY_AND_INFORMATION_CONTRACT'}
    dump(out / 'suite_status.json', status)
    print('RESULT_DIR ' + str(out), flush=True)
    build_compat(ROOT, out / 'compat')
    for idx, name in enumerate(('dasdg', 'dgcrn'), 1):
        print(f'JOB_START {idx}/2 {name}; limited 3 epochs; timeout={args.timeout_per_model}s', flush=True)
        command = [sys.executable, '-B', str(ROOT / 'pilot.py'), '--model', name,
                   '--compat-dir', str(out / 'compat'), '--data-root', str(args.data_root.resolve()),
                   '--out', str(out / name), '--device', args.device]
        status['current_job'] = name
        dump(out / 'suite_status.json', status)
        started = time.monotonic()
        child_env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        try:
            # Inherit stdout so tail -f shows batches while the job is alive.
            process = subprocess.run(command, timeout=args.timeout_per_model, env=child_env)
            marker = out / name / 'PILOT_DONE.json'
            if process.returncode or not marker.is_file():
                raise RuntimeError(f'{name} exit={process.returncode}; PILOT_DONE missing or worker failed')
            done = json.loads(marker.read_text(encoding='utf-8'))
            if done['status'] != 'PILOT_COMPLETE' or not done['pilot_only']:
                raise RuntimeError('Incorrect pilot completion marker')
            for rel, expected in done['files'].items():
                p = out / name / rel
                if not p.resolve().is_relative_to((out / name).resolve()) or sha256(p) != expected:
                    raise ValueError('Pilot output hash mismatch: ' + rel)
            status['models'][name] = {'status': 'PASS_LIMITED_PILOT', 'seconds': time.monotonic() - started,
                                      'completion_marker_sha256': sha256(marker)}
            dump(out / 'suite_status.json', status)
        except Exception as exc:
            status['models'][name] = {'status': 'FAIL_OR_TIMEOUT', 'error': repr(exc),
                                      'seconds': time.monotonic() - started}
            status['status'] = 'FAIL_PILOT'
            status['traceback'] = traceback.format_exc()
            dump(out / 'suite_status.json', status)
            print('FAIL_PILOT ' + name + '; evidence preserved; no automatic retry', flush=True)
            return 2
    status['status'] = 'PASS_LIMITED_PILOT_PENDING_FORMAL_CONTRACT'
    dump(out / 'suite_status.json', status)
    # Checkpoints are local-only pilot artifacts; portable return archive contains
    # diagnostics, configs, compatibility records, and their hashes, not .pt files.
    returned_hashes = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob('*'))
                       if p.is_file() and p.suffix != '.pt'}
    dump(out / 'return_artifact_hashes.json', returned_hashes)
    import tarfile
    archive = out.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as bundle:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix != '.pt':
                bundle.add(p, arcname=out.name + '/' + str(p.relative_to(out)))
    print(status['status'], flush=True)
    print('RETURN_ARCHIVE ' + str(archive), flush=True)
    print('FORMAL_TRAINING_COMPLETED=false; TEST_EVALUATION_PERFORMED=false', flush=True)
    return 0


def main():
    if '--validate-only' in sys.argv:
        return run_suite()
    import fcntl
    with (ROOT / '.pilot_run.lock').open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('ANOTHER_PILOT_IS_RUNNING; second coordinator refused', flush=True)
            return 2
        try:
            return run_suite()
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


if __name__ == '__main__':
    raise SystemExit(main())
