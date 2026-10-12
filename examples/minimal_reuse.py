"""Self-contained synthetic example; these numbers are not paper results."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new example directory')
    args.output.mkdir(parents=True)
    import numpy as np
    from dasdg.api import Config, fit, predict
    t = np.arange(160)[:, None]
    flow = (20 + 3*np.sin(t/8 + np.arange(5)[None]/4)).astype(np.float32)
    mask = np.ones(flow.shape, bool)
    mask[13:24, 1] = False
    mask[127:139, 2] = False
    data = args.output / 'dataset.npz'
    np.savez_compressed(data, flow=flow, observed=mask, split=np.array([80,120,160]))
    config = Config(input_steps=6, output_steps=4, hidden_dim=8, embedding_dim=4,
                    graph_dim=4, top_k=3, batch_size=16, epochs=2, patience=10, seed=123)
    (args.output / 'config.json').write_text(json.dumps(asdict(config), indent=2))
    fit(data, args.output/'train', config, device=args.device)
    values = flow[120:144].copy()
    observed = mask[120:144]
    values[~observed] = np.nan
    np.savez_compressed(args.output/'history.npz', values=values, observed=observed)
    pred = predict(args.output/'train/best.pt', values, observed, args.device)
    np.save(args.output/'prediction.npy', pred, allow_pickle=False)
    print(json.dumps({'scope': 'synthetic_example_not_benchmark', 'prediction_shape': list(pred.shape)}))


if __name__ == '__main__':
    main()
