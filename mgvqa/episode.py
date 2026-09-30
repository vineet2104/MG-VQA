"""
core.episode — Episode lifecycle: seeding, scene setup, and scheduling.

This module provides the deterministic seeding logic and episode scheduling
that ensures reproducibility between agentic and singlepass runs.

CRITICAL INVARIANT:
    episode_seed = base_seed + 1000 * ep_index
    (+ retry_offset * 137 for scene retries)

    Same seed + same pool → identical scenes/questions/options across modes.
"""

import os
import json
import random
import numpy as np
from typing import List, Optional, Tuple
from collections import Counter


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def compute_episode_seed(base_seed: int, ep_index: int, retry: int = 0) -> int:
    """Compute the deterministic episode seed.

    Args:
        base_seed: Run-level seed (e.g. 1000000 for eval benchmark).
        ep_index: Episode index (0-based).
        retry: Scene retry count (for insufficient occlusion retries).

    Returns:
        Integer seed for this episode.
    """
    return base_seed + 1000 * ep_index + retry * 137


def setup_episode_rng(episode_seed: int) -> random.Random:
    """Create a per-episode RNG and seed global state.

    Seeds both the returned Random instance AND the global random/numpy
    RNGs (needed by PyBullet spawning which uses global state).

    Returns:
        A seeded random.Random instance for episode-local decisions.
    """
    random.seed(episode_seed)
    np.random.seed(episode_seed)
    return random.Random(episode_seed)


# ---------------------------------------------------------------------------
# Episode scheduling
# ---------------------------------------------------------------------------

QUESTION_TYPES = [
    "counting",
    "interactive_counting",
    "existence",
    "same_attribute",
    "relative_size",
    "under_object",
]

# Benchmark V2 (branch: benchmark-v2) — kept in sync with
# questions.QUESTION_TYPES_V2.  See docs/BENCHMARK_V2_DESIGN.md.
# `rearrangement` was dropped from the default pool (still runnable via
# --question_type rearrangement).  `compare` (marked three-way size
# comparison) was added as the 4th V2 task.  MUST stay in sync with
# questions.QUESTION_TYPES_V2 (the authoritative list the worker schedules from).
QUESTION_TYPES_V2 = [
    "count",
    "find",
    "beneath",
    "compare",
]


def build_episode_schedule(
    n_episodes: int,
    question_type: str,
    base_seed: int,
) -> List[str]:
    """Build a deterministic episode-to-question-type schedule.

    When question_type is "random", episodes are distributed equally
    across all 6 types and shuffled deterministically with base_seed.
    Otherwise, all episodes use the specified type.

    The schedule is ALWAYS built from the full 6-type list even in
    singlepass mode, ensuring seed-identical schedules between modes.

    Args:
        n_episodes: Total number of episodes.
        question_type: "random" or a specific question type.
        base_seed: Seed for deterministic shuffling.

    Returns:
        List of question type strings, length n_episodes.
    """
    if question_type == "random":
        all_types = list(QUESTION_TYPES)  # deterministic order
        n_per_type = -(-n_episodes // len(all_types))  # ceil division
        schedule = []
        for qt in all_types:
            schedule.extend([qt] * n_per_type)
        schedule = schedule[:n_episodes]
        random.Random(base_seed).shuffle(schedule)
    else:
        schedule = [question_type] * n_episodes

    return schedule


def print_schedule_summary(schedule: List[str]) -> None:
    """Print a breakdown of episode counts per question type."""
    n_per_qt = Counter(schedule)
    print(f"\n  Episode schedule ({len(schedule)} total):")
    for qt in QUESTION_TYPES:
        if n_per_qt[qt]:
            print(f"    {qt:<20s} {n_per_qt[qt]:>4d} episodes")
    print()


# ---------------------------------------------------------------------------
# Episode subset loading
# ---------------------------------------------------------------------------

def load_episode_subset(subset_path: str) -> set:
    """Load episode indices from a JSON subset file.

    Expected format: {"episodes": [0, 1, 5, 7, ...], ...}

    Returns:
        Set of allowed episode indices.
    """
    with open(subset_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    return {int(e) for e in doc.get("episodes", [])}


# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------

def create_episode_dir(save_dir: str, ep_index: int, question_type: str) -> str:
    """Create and return the episode output directory.

    Format: save_dir/ep_NNN_question_type/
    """
    ep_dirname = f"ep_{ep_index:03d}_{question_type}"
    ep_dir = os.path.join(save_dir, ep_dirname)
    os.makedirs(ep_dir, exist_ok=True)
    return ep_dir


def save_selected_objects(
    ep_dir: str,
    ep_index: int,
    episode_seed: int,
    question_type: str,
    obj_names: List[str],
    obj_folders: List[str],
) -> None:
    """Save selected_objects.json for reproducibility."""
    with open(os.path.join(ep_dir, "selected_objects.json"), "w", encoding="utf-8") as f:
        json.dump({
            "episode": ep_index,
            "episode_seed": episode_seed,
            "question_type": question_type,
            "selected_obj_names": obj_names,
            "selected_obj_folders": obj_folders,
        }, f, indent=2)
