#!/usr/bin/env python3
"""Local dependency checks. No API calls or simulator scene mutation."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/graspgen'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['direct', 'perception', 'agentic'], default='agentic')
    p.add_argument('--snapshot', type=Path, default=ROOT / 'approved-questions/benchmark_600')
    args = p.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    failures = []

    def check(name, fn):
        try:
            detail = fn()
            print(f'OK   {name}: {detail}')
        except Exception as exc:
            failures.append(name)
            print(f'FAIL {name}: {exc}')

    modules = ['numpy', 'cv2', 'pybullet', 'yaml', 'tools', 'mgvqa.worker']
    if args.mode != 'direct':
        modules += ['ultralytics']
    if args.mode == 'agentic':
        modules += ['pointnet2_ops', 'grasp_gen.grasp_server']
    for module in modules:
        check(module, lambda m=module: importlib.import_module(m).__file__)

    def gpu():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        # Execute a kernel, not just a device-query check.
        (torch.ones(8, device='cuda') + 1).sum().item()
        return f'{torch.cuda.get_device_name(0)}; torch={torch.__version__}'
    check('GPU kernel', gpu)

    from robo_blender.server_launcher import _find_blender
    check('Blender executable', _find_blender)

    def bank():
        files = sorted(args.snapshot.glob('ep_*/scene_snapshot/scene_snapshot.json'))
        if len(files) != 600:
            raise RuntimeError(f'Expected 600 snapshots, found {len(files)}')
        for f in files:
            json.loads(f.read_text())
        return f'600 readable snapshots in {args.snapshot} (not a semantic audit)'
    check('Bench bank', bank)

    paths = []
    if args.mode != 'direct':
        paths.append('third_party/sam3/sam3.pt')
    if args.mode == 'agentic':
        paths += ['checkpoints/graspgen/graspgen_robotiq_2f_140_gen.pth',
                  'checkpoints/graspgen/graspgen_robotiq_2f_140_dis.pth']
    for path in paths:
        def exists(path=path):
            if not (ROOT / path).is_file():
                raise FileNotFoundError(path)
            return 'present'
        check(path, exists)
    print('NVIDIA credential:', 'set' if os.getenv('NV_API_KEY') or os.getenv('NVIDIA_API_KEY') else 'not set')
    print('Preflight does not verify rendering, API access, grasp quality, or a fresh installation.')
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
