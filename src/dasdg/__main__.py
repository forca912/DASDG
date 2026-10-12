"""Unified CLI. Archived benchmark commands retain their own frozen protocols."""
import argparse
import json
import sys


def main(argv=None):
    from . import __version__
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {'train', 'predict'}:
        from dasdg_reuse.__main__ import main as reuse_main
        old = sys.argv
        try:
            sys.argv = ['dasdg'] + argv
            return reuse_main()
        finally:
            sys.argv = old
    if argv and argv[0] == 'evidence':
        from dasdg_revision_tools.__main__ import main as evidence_main
        return evidence_main(argv[1:])
    parser = argparse.ArgumentParser(prog='dasdg', description='DASDG: train, predict, evidence and external-source compatibility')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('status')
    for name in ('train', 'predict', 'evidence'):
        sub.add_parser(name, add_help=False)
    command = sub.add_parser('dgcrn-prepare', help='Verify user-supplied pinned upstream and create an isolated compatibility copy')
    command.add_argument('--upstream', required=True)
    command.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.action == 'status':
        print(json.dumps({'version': __version__, 'distribution_kind': 'release',
            'api_models': ['dasdg', 'gwn_duration', 'gru_duration'],
            'api_inputs': 'finite_complete_training_labels_and_boolean_observation_mask',
            'api_resume': 'epoch_boundary_optimizer_scheduler_RNG_restore',
            'api_protocol': 'custom_reuse_not_frozen_PEMS_benchmark',
            'frozen_workflows': 'source_bundle/frozen; original protocol identities retained',
            'dgcrn': 'external_adapted_baseline_not_native_DASDG_API; upstream_not_bundled',
            'dgcrn_license': 'MIT_file_vs_CC_BY_NC_SA_README_conflict_unresolved',
            'gpu_acceptance': 'requires_actual_execution; not_implied_by_CPU_pass'}, indent=2))
    elif args.action == 'dgcrn-prepare':
        from .external_dgcrn import prepare
        print(json.dumps(prepare(args.upstream, args.output), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
