"""Shared, collision-safe experiment directory naming."""
from datetime import datetime
from pathlib import Path
import re


def create_run_directory(model, mode, root='rollouts', timestamp=None):
    """Create a new run; never merge separate experiments, even in one second."""
    timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    mode = 'direct' if mode == 'singlepass' else mode
    def slug(value):
        return re.sub(r'[^A-Za-z0-9._-]+', '-', str(value)).strip('.-_') or 'unknown'
    stem = f'{slug(model)}_{slug(mode)}_{timestamp}'
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        path = root / (stem if index == 0 else f'{stem}_{index:02d}')
        try:
            path.mkdir()  # atomic reservation, including concurrent launches
            return str(path)
        except FileExistsError:
            index += 1
