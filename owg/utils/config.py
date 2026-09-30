"""Minimal config loader (kept for backward compatibility with demo scripts)."""

import yaml
from easydict import EasyDict as edict


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = edict(config)
    return config
