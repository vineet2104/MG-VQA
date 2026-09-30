#!/usr/bin/env python3
"""Download only the 600-question MG-VQA-Bench bank; never the Hard bank."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='bhatvineet/mg-vqa-bench',
                   help='Public Hugging Face dataset ID')
    p.add_argument('--revision', default='78c4d8adf7a5c68060bc32b340c8a746933f659c',
                   help='Pinned dataset commit SHA')
    p.add_argument('--output', type=Path, default=Path('approved-questions/mg-vqa-bench'))
    args = p.parse_args()
    if 'hard' in args.repo.lower():
        p.error('This release downloader only accepts the Bench bank, not Hard.')
    snapshot_download(repo_id=args.repo, repo_type='dataset', revision=args.revision,
                      local_dir=str(args.output))
    snapshots = list(args.output.glob('ep_*/scene_snapshot/scene_snapshot.json'))
    if len(snapshots) != 600:
        raise RuntimeError(f'Expected 600 snapshots at {args.output}; found {len(snapshots)}. '
                           'Check the dataset layout before evaluation.')
    print(f'600 snapshots downloaded to {args.output}')


if __name__ == '__main__':
    main()
