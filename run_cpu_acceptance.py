"""Short software acceptance suite. All generated data are synthetic fixtures.

Runs against the INSTALLED package from a directory outside its source tree.
Includes real train/interruption/resume, weight reload, API/CLI and rejection tests.
No traffic benchmark result is generated or altered.
"""
from __future__ import annotations
import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Acceptance output must be empty; do not overwrite earlier evidence')
    out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'RUNNING', 'scope': 'synthetic_software_acceptance_not_paper_benchmark',
              'checks': {}, 'package': {}, 'device': 'cpu'}
    report_path = out / 'acceptance.json'

    def record(name, details=True):
        report['checks'][name] = details
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print('PASS', name, flush=True)

    def rejected(name, callback, exception):
        try:
            callback()
        except exception as exc:
            record(name, {'rejected_as': type(exc).__name__, 'message': str(exc)})
        else:
            raise AssertionError(f'{name}: invalid operation was accepted')

    try:
        import numpy as np
        import torch
        import dasdg_reuse
        from dasdg_reuse.api import Config, fit, predict, load_dataset, source_identity
        package_path = Path(dasdg_reuse.__file__).resolve()
        if 'site-packages' not in str(package_path):
            raise AssertionError(f'Must test installed wheel, not source tree: {package_path}')
        report['package'] = {'version': dasdg_reuse.__version__, 'installed_path': str(package_path),
                             'python': platform.python_version(), 'torch': str(torch.__version__),
                             'numpy': str(np.__version__), 'source_sha256': source_identity()}
        record('installed_wheel_import')
        command = [sys.executable, '-m', 'dasdg_reuse', '--version']
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        record('module_cli_version', completed.stdout.strip())
        cli = Path(sys.executable).parent / ('dasdg-reuse.exe' if sys.platform == 'win32' else 'dasdg-reuse')
        completed = subprocess.run([str(cli), '--version'], check=True, text=True, capture_output=True)
        record('console_entrypoint', completed.stdout.strip())
        rng = np.random.default_rng(123)
        flow = (20 + 3*np.sin(np.arange(160)[:, None]/8 + np.arange(5)[None]/4)
                + rng.normal(0, 0.15, (160, 5))).astype(np.float32)
        observed = np.ones(flow.shape, dtype=bool)
        observed[13:24, 1] = False
        observed[75:83, 3] = False
        observed[127:139, 2] = False
        dataset = out / 'synthetic_multichannel.npz'
        np.savez_compressed(dataset, flow=flow, observed=observed, split=np.array([80,120,160],np.int64))
        cfg = Config(input_steps=6, output_steps=4, hidden_dim=8, embedding_dim=4, graph_dim=4,
                     top_k=3, batch_size=16, epochs=3, patience=10, dropout=0.1, graph_dropout=0.1,
                     seed=123, cpu_threads=2)
        (out / 'example_config.json').write_text(json.dumps(asdict(cfg), indent=2), encoding='utf-8')
        loaded, _, split = load_dataset(dataset)
        assert loaded.shape == (160,5) and split == (80,120,160)
        record('custom_dataset_import')
        full = fit(dataset, out / 'uninterrupted', cfg)
        record('short_training', {'epochs': full['epochs_completed'], 'fixture_only': True})
        assert full['metadata']['mean'] == float(flow[:80].mean(dtype=np.float64))
        assert full['metadata']['std'] == float(flow[:80].std(dtype=np.float64))
        record('normalization_uses_training_labels_only')
        paused = fit(dataset, out / 'resumed', cfg, stop_after_epoch=1)
        assert paused['status'] == 'paused_at_epoch_boundary'
        record('epoch_boundary_interruption')
        resumed = fit(dataset, out / 'resumed', cfg, resume=True)
        a = torch.load(out/'uninterrupted'/'last.pt', map_location='cpu', weights_only=True)
        b = torch.load(out/'resumed'/'last.pt', map_location='cpu', weights_only=True)
        assert full['history'] == resumed['history']
        assert all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
        record('resume_matches_uninterrupted_exactly_cpu', {'epochs': len(resumed['history']), 'weights_exact': True})
        values, mask = flow[120:144].copy(), observed[120:144].copy()
        p = predict(out/'uninterrupted'/'best.pt', values, mask)
        q = predict(out/'resumed'/'best.pt', values, mask)
        assert p.shape == (4,5) and np.isfinite(p).all() and np.array_equal(p,q)
        record('checkpoint_reload_inference', {'shape': list(p.shape), 'resume_predictions_exact': True})
        changed = values.copy()
        changed[~mask] = np.nan
        assert np.array_equal(p, predict(out/'uninterrupted'/'best.pt', changed, mask))
        record('unobserved_values_do_not_affect_predictions')
        history = out/'history_input.npz'
        np.savez_compressed(history, values=values, observed=mask)
        predicted = out/'cli_prediction.npy'
        subprocess.run([sys.executable, '-m', 'dasdg_reuse', 'predict', '--checkpoint',
                        str(out/'uninterrupted'/'best.pt'), '--history', str(history),
                        '--output', str(predicted)], check=True)
        assert np.array_equal(np.load(predicted, allow_pickle=False), p)
        record('cli_prediction_matches_api')
        bad = out/'invalid_mask.npz'
        np.savez_compressed(bad, flow=flow, observed=observed.astype(np.uint8), split=np.array([80,120,160]))
        rejected('reject_nonboolean_mask', lambda: load_dataset(bad), ValueError)
        bad_flow = flow.copy(); bad_flow[15,1] = np.nan
        missing_labels = out/'missing_labels.npz'
        np.savez_compressed(missing_labels, flow=bad_flow, observed=observed, split=np.array([80,120,160]))
        rejected('reject_incomplete_training_labels', lambda: load_dataset(missing_labels), ValueError)
        rejected('reject_checkpoint_node_mismatch', lambda: predict(out/'resumed'/'best.pt', values[:,:4],mask[:,:4]), ValueError)
        rejected('reject_resume_changed_config', lambda: fit(dataset,out/'resumed',replace(cfg,seed=124),resume=True), ValueError)
        changed_data = out/'changed_data.npz'
        changed_flow = flow.copy(); changed_flow[-1,0] += 1
        np.savez_compressed(changed_data,flow=changed_flow,observed=observed,split=np.array([80,120,160]))
        rejected('reject_resume_changed_dataset', lambda: fit(changed_data,out/'resumed',cfg,resume=True), ValueError)
        rejected('reject_output_overwrite', lambda: fit(dataset,out/'resumed',cfg), FileExistsError)
        from dasdg_reuse.api import _run_lock
        with _run_lock(out/'locked'/'.run.lock'):
            rejected('reject_concurrent_output_use', lambda: fit(dataset,out/'locked',cfg), RuntimeError)
        for name in ('gwn_duration', 'gru_duration'):
            result = fit(dataset, out/name, replace(cfg,model=name,epochs=1))
            pred = predict(out/name/'best.pt',values,mask)
            assert result['status']=='complete' and pred.shape==(4,5) and np.isfinite(pred).all()
            record(name+'_same_api_smoke')
        report['status'] = 'PASS'
        report['benchmark_result'] = False
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print('SOFTWARE_ACCEPTANCE_PASS', report_path, flush=True)
    except BaseException as exc:
        report['status'] = 'FAIL'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
