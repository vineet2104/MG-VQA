from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from mgvqa.output import create_run_directory


def test_model_mode_timestamp_and_collision(tmp_path):
    args = ('azure/openai/gpt-6-astra', 'agentic')
    first = Path(create_run_directory(*args, root=tmp_path, timestamp='20260927_183045'))
    second = Path(create_run_directory(*args, root=tmp_path, timestamp='20260927_183045'))
    assert first.name == 'azure-openai-gpt-6-astra_agentic_20260927_183045'
    assert second.name == first.name + '_01'
    assert first.is_dir() and second.is_dir()


def test_safe_paths_and_mode_alias(tmp_path):
    path = Path(create_run_directory('../../custom model', 'singlepass', root=tmp_path))
    assert path.parent == tmp_path
    assert path.name.startswith('custom-model_direct_')


def test_concurrent_experiments_never_share_directory(tmp_path):
    def create(_):
        return create_run_directory('model', 'perception', root=tmp_path,
                                    timestamp='20260927_183045')
    with ThreadPoolExecutor(max_workers=4) as pool:
        dirs = list(pool.map(create, range(8)))
    assert len(set(dirs)) == 8
