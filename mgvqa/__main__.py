"""Public Python CLI: python -m mgvqa (one mode per invocation)."""
import argparse
import shlex
import sys


def episode_selection(spec):
    if spec.lower() == 'all':
        return ''
    if spec == '25x4':
        return ','.join(str(base + i) for base in (0, 150, 300, 450)
                        for i in range(0, 150, 6))
    import re
    if not re.fullmatch(r'\d+(?:-\d+|(?:,\d+)*)', spec):
        raise argparse.ArgumentTypeError('Use all, 25x4, comma-separated indices, or a-b')
    if '-' in spec:
        start, end = map(int, spec.split('-'))
        if end < start:
            raise argparse.ArgumentTypeError('Episode range must be ascending')
    return spec


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('Must be non-negative')
    return number


def positive(value):
    number = nonnegative(value)
    if number == 0:
        raise argparse.ArgumentTypeError('Must be positive')
    return number


def build_args(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate MG-VQA-Bench frozen scenes.')
    parser.add_argument('--mode', choices=['direct', 'perception', 'agentic'], default='agentic')
    parser.add_argument('--model', default='gcp/google/gemini-3.1-pro-preview')
    parser.add_argument('--scene-model', default='gcp/google/gemini-3.6-flash')
    parser.add_argument('--episodes', type=episode_selection, default='all',
                        help=argparse.SUPPRESS)  # optional internal smoke-test subset
    parser.add_argument('--dataset', '--snapshot', dest='dataset',
                        default='approved-questions/benchmark_600',
                        help='Downloaded dataset directory; evaluates every episode (default: %(default)s)')
    parser.add_argument('--workers', type=nonnegative, default=4, help='0 = single process')
    parser.add_argument('--gpus', type=positive, default=1)
    parser.add_argument('--recycle', type=positive, default=4)
    parser.add_argument('--dry-run', action='store_true', help='Print command without loading models or calling APIs')
    args = parser.parse_args(argv)
    from pathlib import Path
    args.dataset = str(Path(args.dataset).expanduser().resolve())
    runtime_args = ['--mode', args.mode, '--model_name', args.model,
                    '--scene_model', args.scene_model, '--from_snapshot', args.dataset,
                    '--n_workers', str(args.workers),
                    '--max_eps_per_subprocess', str(args.recycle if args.workers else 0)]
    # GPU distribution/recycling belong to the orchestrator; a single process
    # must not exit early after the first recycle-sized chunk.
    if args.workers:
        runtime_args += ['--n_gpus', str(args.gpus)]
    if args.episodes:
        runtime_args += ['--episodes', args.episodes]
    return args, runtime_args


def main():
    args, runtime_args = build_args()
    if args.dry_run:
        print(shlex.join([sys.executable, '-m', 'mgvqa.run_benchmark', *runtime_args]))
        return
    import os
    os.environ.setdefault('MALLOC_ARENA_MAX', '2')
    from mgvqa.run_benchmark import main as run
    sys.argv = ['mgvqa.run_benchmark', *runtime_args]
    run()


if __name__ == '__main__':
    main()
