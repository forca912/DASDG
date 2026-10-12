"""Command line interfaces for the candidate software API."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog='dasdg-reuse')
    parser.add_argument('--version', action='version', version='0.1.0rc1')
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--dataset', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--config', type=Path, required=True)
    train.add_argument('--device', default='cpu')
    train.add_argument('--resume', action='store_true')
    train.add_argument('--stop-after-epoch', type=int)
    infer = sub.add_parser('predict')
    infer.add_argument('--checkpoint', type=Path, required=True)
    infer.add_argument('--history', type=Path, required=True)
    infer.add_argument('--output', type=Path, required=True)
    infer.add_argument('--device', default='cpu')
    args = parser.parse_args()
    from .api import Config, fit, predict
    if args.command == 'train':
        cfg = Config(**json.loads(args.config.read_text(encoding='utf-8')))
        result = fit(args.dataset, args.output, cfg, args.device, args.resume, args.stop_after_epoch)
        print(json.dumps({k: v for k, v in result.items() if k not in ('metadata', 'history')}, indent=2))
    else:
        import numpy as np
        if args.output.exists():
            raise FileExistsError(f'Refusing overwrite: {args.output}')
        with np.load(args.history, allow_pickle=False) as z:
            result = predict(args.checkpoint, z['values'], z['observed'], args.device)
        with args.output.open('wb') as stream:
            np.save(stream, result, allow_pickle=False)
        print(json.dumps({'prediction_shape': list(result.shape), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
