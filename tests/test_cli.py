"""Public command surface: dry runs never need simulator/model imports."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
from mgvqa.__main__ import build_args, episode_selection

ROOT = Path(__file__).resolve().parents[1]


def test_single_process_has_no_orchestrator_gpu_flag_or_early_recycle():
    _, args = build_args(['--workers', '0', '--episodes', '0,150,300,450'])
    assert '--n_gpus' not in args
    assert args[args.index('--max_eps_per_subprocess') + 1] == '0'
    assert args[args.index('--episodes') + 1] == '0,150,300,450'


def test_parallel_forwarding_and_subset():
    _, args = build_args(['--workers', '2', '--gpus', '3', '--recycle', '4',
                          '--model', 'custom/planner', '--scene-model', 'custom/scene',
                          '--episodes', '25x4'])
    assert args[args.index('--n_gpus') + 1] == '3'
    assert args[args.index('--model_name') + 1] == 'custom/planner'
    assert args[args.index('--scene_model') + 1] == 'custom/scene'
    eps = list(map(int, args[args.index('--episodes') + 1].split(',')))
    assert len(eps) == 100 and len(set(eps)) == 100
    assert [sum(b <= i < b + 150 for i in eps) for b in (0, 150, 300, 450)] == [25] * 4
    assert episode_selection('all') == ''


@pytest.mark.parametrize('args', [[], ['--mode', 'direct'], ['--mode', 'perception']])
def test_python_dry_run(args):
    result = subprocess.run([sys.executable, '-m', 'mgvqa', *args, '--dry-run'],
                            cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '-m mgvqa.run_benchmark' in result.stdout


def test_shell_mode_loop_from_another_directory(tmp_path):
    result = subprocess.run([str(ROOT / 'run_mgvqa.sh'), '--mode',
                             'direct,perception,agentic', '--workers', '0', '--dry-run'],
                            cwd=tmp_path, env={**os.environ, 'PYTHON': sys.executable},
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('-m mgvqa.run_benchmark') == 3
    assert '--n_gpus' not in result.stdout


@pytest.mark.parametrize('args', [['--mode', 'bad'], ['--mode'], ['--mode', 'direct,'],
                                 ['--workers', '-1'], ['--gpus', '0'],
                                 ['--episodes', '4-1'], ['--model']])
def test_shell_rejects_invalid_options(args):
    result = subprocess.run([str(ROOT / 'run_mgvqa.sh'), *args, '--dry-run'],
                            env={**os.environ, 'PYTHON': sys.executable},
                            text=True, capture_output=True)
    assert result.returncode != 0


@pytest.mark.parametrize('workers', [0, 2])
def test_public_flags_parse_in_runtime(workers):
    from unittest.mock import patch
    from mgvqa.run_benchmark import _pin_eval_defaults
    from mgvqa.cli import parse_args as worker_parse
    from mgvqa.orchestrator.parallel import parse_args as parallel_parse
    _, args = build_args(['--workers', str(workers), '--episodes', '0,150,300,450'])
    with patch('mgvqa.scene_snapshot.discover_snapshot_episodes', return_value=list(range(600))):
        argv = _pin_eval_defaults(['mgvqa.run_benchmark', *args])
    with patch.object(sys, 'argv', argv):
        parsed = parallel_parse() if workers else worker_parse()
    assert (parsed.target_episodes if workers else parsed.episodes) == '0,150,300,450'
    assert parsed.n_episodes == 600
    assert parsed.episode_subset is None


def test_dataset_path_runs_entire_bank_without_subset(tmp_path):
    dataset = tmp_path / 'downloaded dataset'
    dataset.mkdir()
    _, args = build_args(['--dataset', str(dataset), '--model', 'custom/model',
                          '--mode', 'perception', '--workers', '4'])
    assert args[args.index('--from_snapshot') + 1] == str(dataset)
    assert '--episodes' not in args
    _, alias = build_args(['--snapshot', str(dataset)])
    assert alias[alias.index('--from_snapshot') + 1] == str(dataset)


@pytest.mark.parametrize('style', ['separate', 'equals'])
def test_shell_dataset_is_relative_to_caller(tmp_path, style):
    import shlex
    flag = ['--dataset', 'my dataset'] if style == 'separate' else ['--dataset=my dataset']
    result = subprocess.run([str(ROOT / 'run_mgvqa.sh'), *flag, '--dry-run'],
                            cwd=tmp_path, env={**os.environ, 'PYTHON': sys.executable},
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[command.index('--from_snapshot') + 1] == str(tmp_path / 'my dataset')
    assert '--episodes' not in command


@pytest.mark.parametrize('mode', ['direct', 'singlepass', 'perception', 'agentic'])
def test_model_banner_only_describes_active_components(mode, capsys):
    from types import SimpleNamespace
    from mgvqa.orchestrator.parallel import print_model_banner
    print_model_banner(SimpleNamespace(mode=mode, model_name='planner',
                                       scene_model='scene-backend'))
    text = capsys.readouterr().out
    assert 'Pointing VLM' not in text
    assert ('scene-backend' in text) == (mode in {'perception', 'agentic'})
    if mode in {'direct', 'singlepass'}:
        assert 'disabled (one image, no tools)' in text
        assert 'SAM3' not in text


def test_worker_script_default_is_new_namespace():
    from mgvqa.orchestrator.parallel import parse_args, build_worker_cmd
    from unittest.mock import patch
    with patch.object(sys, 'argv', ['test', '--n_episodes', '600']):
        args = parse_args()
    with patch.dict(os.environ, {}, clear=True):
        cmd = build_worker_cmd(args, 0, [0], '/tmp/worker', '/tmp/pause')
    assert cmd[0] == sys.executable
    assert Path(cmd[1]) == ROOT / 'mgvqa' / 'worker_entry.py'
    assert Path(cmd[1]).is_file()
