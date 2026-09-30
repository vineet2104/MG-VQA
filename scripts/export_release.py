#!/usr/bin/env python3
"""Export a reviewable clean tree without .git/history, datasets or local logs.

Dry-run by default. Includes working-tree changes, not just HEAD. Destination
must be a new directory outside this repo. Nothing is committed or pushed.
"""
import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOTS = {'mgvqa', 'models', 'perception', 'manipulation', 'owg', 'owg_robot',
                 'robo_blender', 'experiments', 'scene_gen', 'config', 'checkpoints',
                 'third_party', 'scripts', 'tests'}
ALLOWED_FILES = {'README.md', 'models.md', 'INSTALL.md', 'ARCHITECTURE.md',
                 'environment.yml', 'requirements-runtime.txt', '.env.example',
                 '.gitignore', '.gitattributes', 'run_mgvqa.sh', 'tools.py',
                 'questions.py', 'gt_visibility_filter.py', 'QUESTIONS.MD',
                 'sim-bench-examples.png', 'LICENSE'}


def candidates():
    output = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others',
                                      '--exclude-standard'], cwd=ROOT)
    files = sorted(set(p.decode() for p in output.split(b'\0') if p))
    # --no-index intentionally honors ignore rules even for tracked old scripts.
    proc = subprocess.run(['git', 'check-ignore', '--no-index', '-z', '--stdin'],
                          input=b'\0'.join(f.encode() for f in files)+b'\0',
                          stdout=subprocess.PIPE, cwd=ROOT)
    if proc.returncode not in (0, 1):
        raise RuntimeError('git check-ignore failed')
    ignored = set(p.decode() for p in proc.stdout.split(b'\0') if p)
    return [f for f in files if f not in ignored and (ROOT/f).is_file()
            and (f in ALLOWED_FILES or Path(f).parts[0] in ALLOWED_ROOTS)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    files = candidates()
    for f in files:
        print(f)
    print(f'{len(files)} files; third-party license review still required.')
    if not args.apply:
        return
    if not args.output:
        p.error('--apply requires --output')
    dst = args.output.resolve()
    if dst == ROOT or ROOT in dst.parents:
        p.error('Destination must be outside the working repository')
    if dst.exists():
        p.error('Destination already exists; refusing to overwrite')
    dst.mkdir(parents=True)
    for f in files:
        src = ROOT/f
        if src.is_symlink():
            raise RuntimeError(f'Review symlink before export: {f}')
        target = dst/f
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    print(f'Exported to {dst}; no .git copied and no remote configured.')


if __name__ == '__main__':
    main()
