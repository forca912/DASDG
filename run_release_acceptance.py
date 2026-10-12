"""Installed-wheel integration tests; GPU checks require actual CUDA hardware."""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def same_state(a, b, path='state'):
    import torch
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape:
            raise AssertionError('Checkpoint tensor structure differs: '+path)
        if not torch.equal(a.cpu(), b.cpu()):
            difference = float((a.cpu().double()-b.cpu().double()).abs().max())
            raise AssertionError(f'Checkpoint tensor differs: {path}; max_abs_difference={difference}')
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise AssertionError('Checkpoint keys differ: '+path)
        for key in a:
            same_state(a[key], b[key], path+'.'+str(key))
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise AssertionError('Checkpoint sequence differs: '+path)
        for i, (x, y) in enumerate(zip(a, b)):
            same_state(x, y, path+f'[{i}]')
    elif a != b:
        raise AssertionError(f'Checkpoint value differs: {path}; {a!r} vs {b!r}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--external-upstream', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError('Choose a new integration acceptance output')
    out.mkdir(parents=True)
    package = Path(__file__).resolve().parent
    report = {'status': 'RUNNING', 'scope': 'synthetic_software_acceptance_not_benchmark',
              'checks': {}, 'cuda_requested': args.cuda}

    def record(name, detail=True):
        report['checks'][name] = detail
        (out/'acceptance.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print('PASS', name, flush=True)

    try:
        import numpy as np
        import torch
        import dasdg
        from dasdg.api import Config, fit, predict
        from dasdg_reuse.api import Config as original_config, fit as original_fit, predict as original_predict
        assert Config is original_config and fit is original_fit and predict is original_predict
        assert importlib.metadata.version('dasdg') == dasdg.__version__ == '1.1.0'
        assert 'site-packages' in str(Path(dasdg.__file__).resolve())
        report['environment'] = {'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': str(np.__version__), 'cuda_runtime': torch.version.cuda,
            'installed_path': str(Path(dasdg.__file__).resolve()), 'distribution_version': dasdg.__version__,
            'torch_dependency_install': 'reused_existing_runtime_not_from_zero'}
        record('unified_installed_wheel_and_API_identity')
        console = Path(sys.executable).parent / ('dasdg.exe' if os.name == 'nt' else 'dasdg')
        result = subprocess.run([str(console), '--version'], check=True, text=True, capture_output=True)
        assert result.stdout.strip() == '1.1.0'
        record('unified_console_entrypoint')
        result = subprocess.run([sys.executable, '-m', 'dasdg', 'status'], check=True, text=True, capture_output=True)
        status = json.loads(result.stdout)
        assert status['distribution_kind']=='release'
        record('unified_status_and_scope', status)
        provenance = json.loads((package/'source_provenance.json').read_text(encoding='utf-8'))
        for entry in provenance['records']:
            p = package / entry['bundled_path']
            assert hashlib.sha256(p.read_bytes()).hexdigest() == entry['sha256'], entry['bundled_path']
            if entry['bundled_path'].startswith('src/'):
                installed = Path(dasdg.__file__).resolve().parent.parent / entry['bundled_path'][4:]
                assert hashlib.sha256(installed.read_bytes()).hexdigest() == entry['sha256'], str(installed)
        record('frozen_sources_and_installed_audited_API_unchanged', len(provenance['records']))
        result = subprocess.run([sys.executable, '-m', 'dasdg', 'evidence', 'registry'], check=True, text=True, capture_output=True)
        assert json.loads(result.stdout)['tools_version']=='0.3.0'
        record('integrated_evidence_CLI')
        # These calls display help only; they cannot start a formal experiment.
        for module in ('dasdg.run_dasdg_reproduction', 'dasdg.train_fdr_stgnn', 'dasdg.train_v16b_external'):
            subprocess.run([sys.executable, '-m', module, '--help'], check=True, capture_output=True)
        record('archived_entrypoints_import_and_help')
        subprocess.run([sys.executable, str(package/'examples/minimal_reuse.py'), '--output', str(out/'example')], check=True)
        assert np.load(out/'example/prediction.npy', allow_pickle=False).shape == (4,5)
        record('documented_example_runs_outside_source_tree')
        from dasdg.external_dgcrn import prepare
        # Error checks need no third-party source in the bundle.
        try:
            prepare(out/'nonexistent_upstream', out/'must_not_exist')
        except ValueError:
            assert not (out/'must_not_exist').exists()
            record('external_source_missing_is_rejected_before_output')
        else:
            raise AssertionError('Missing external source was accepted')
        if args.external_upstream:
            prepared = prepare(args.external_upstream, out/'compatibility')
            original = (args.external_upstream/'methods/DGCRN/net.py').read_text(encoding='utf-8')
            expected = original.replace('from layer import *','from pf_dgcrn.layer import *',1).replace('torch.squeeze(x[..., i])','x[..., i]',1)
            assert (out/'compatibility/pf_dgcrn/net.py').read_bytes()==expected.encode('utf-8')
            record('external_compatibility_same_as_formal_v3', prepared)
        else:
            report['external_operator_acceptance'] = 'not_executed_no_upstream_supplied; source_not_bundled'
        if args.cuda:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; GPU acceptance cannot pass on CPU')
            if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
                raise RuntimeError('Run with CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic CUDA checks')
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats()
            report['environment']['gpu_name'] = torch.cuda.get_device_name(0)
            report['environment']['deterministic_algorithms'] = True
            report['environment']['tf32_enabled'] = False
            report['environment']['cublas_workspace_config'] = os.environ['CUBLAS_WORKSPACE_CONFIG']
            record('actual_CUDA_environment')
            fixture = out/'example/dataset.npz'
            config = Config(input_steps=6, output_steps=4, hidden_dim=8, embedding_dim=4, graph_dim=4,
                            top_k=3, batch_size=16, epochs=3, patience=10, seed=123)
            full = fit(fixture, out/'cuda_full', config, device='cuda')
            paused = fit(fixture, out/'cuda_resumed', config, device='cuda', stop_after_epoch=1)
            assert paused['status']=='paused_at_epoch_boundary'
            resumed = fit(fixture, out/'cuda_resumed', config, device='cuda', resume=True)
            raw = torch.load(out/'cuda_full/last.pt', weights_only=True)
            assert raw['metadata']['device']=='cuda' and any(v.is_cuda for v in raw['model'].values())
            assert torch.cuda.max_memory_allocated()>0
            record('training_checkpoint_was_actually_on_GPU', {'epochs': full['epochs_completed'],
                'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
            a = torch.load(out/'cuda_full/last.pt', map_location='cpu', weights_only=True)
            b = torch.load(out/'cuda_resumed/last.pt', map_location='cpu', weights_only=True)
            same_state(a, b)
            assert full==resumed
            record('CUDA_resume_all_checkpoint_states_exact', {'model_optimizer_scheduler_RNG_history_best_weights': 'exact',
                'max_abs_difference': 0.0, 'scope': 'same_recorded_environment_only'})
            with np.load(out/'example/history.npz', allow_pickle=False) as z:
                values, observed = z['values'], z['observed']
            pred = predict(out/'cuda_full/best.pt', values, observed, device='cuda')
            other = predict(out/'cuda_resumed/best.pt', values, observed, device='cuda')
            assert pred.shape==(4,5) and np.isfinite(pred).all() and np.array_equal(pred,other)
            np.save(out/'cuda_prediction.npy', pred, allow_pickle=False)
            record('CUDA_checkpoint_reload_and_resume_prediction_exact')
            clean = values.copy(); clean[~observed] = 987654.0
            assert np.array_equal(pred, predict(out/'cuda_full/best.pt',clean,observed,device='cuda'))
            record('CUDA_hidden_history_invariance')
            # Explicitly match deterministic settings in the new CLI process.
            launcher = ('import torch,runpy; torch.use_deterministic_algorithms(True); '
                'torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False; '
                'torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False; '
                'runpy.run_module("dasdg",run_name="__main__")')
            subprocess.run([sys.executable, '-c', launcher, 'predict', '--checkpoint', str(out/'cuda_full/best.pt'),
                '--history', str(out/'example/history.npz'), '--output', str(out/'cuda_cli_prediction.npy'), '--device','cuda'], check=True)
            assert np.array_equal(pred, np.load(out/'cuda_cli_prediction.npy',allow_pickle=False))
            record('CUDA_unified_CLI_prediction_exact')
            for model in ('gwn_duration','gru_duration'):
                result = fit(fixture,out/model,replace(config,model=model,epochs=1),device='cuda')
                p = predict(out/model/'best.pt',values,observed,device='cuda')
                assert result['status']=='complete' and p.shape==(4,5) and np.isfinite(p).all()
                record(model+'_CUDA_train_and_predict')
            report['gpu_status']='PASS'
        else:
            report['gpu_status']='NOT_TESTED_LOCAL_CPU_ONLY'
        report['status']='PASS'
        record('integration_acceptance_complete')
        print('RELEASE_ACCEPTANCE_PASS', out/'acceptance.json', flush=True)
    except BaseException as exc:
        report['status']='FAIL'
        report['error']={'type': type(exc).__name__, 'message': str(exc)}
        (out/'acceptance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
