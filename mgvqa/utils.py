"""
core.utils — Shared utility classes and functions.

Extracted from demo_conversational.py for reuse across the codebase.
"""

import os
import cv2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ACTIONS_PER_SCENE = 10   # Max manipulation actions (grasp + push) per episode
VIDEO_OUTPUT_FPS = 10

# V2 shared occlusion bar: a "hidden" target must render below this visibility
# fraction (i.e. occlusion ≥ 1 − this) to count as genuinely occluded.  Used
# by beneath (under_object hidden-visibility gate) AND find-yes so both tasks
# demand the SAME degree of occlusion of the queried target.
V2_OCCLUDED_MAX_VISIBILITY = 0.30

from mgvqa.prompts import load_system_prompt
SINGLEPASS_SYSTEM_PROMPT = load_system_prompt("direct")


# ---------------------------------------------------------------------------
# Tee — dual output to terminal + log file
# ---------------------------------------------------------------------------

class Tee:
    """Write to multiple file-like objects simultaneously."""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        if not isinstance(data, str):
            data = str(data)
        for f in self.files:
            f.write(data)

    def flush(self):
        for f in self.files:
            f.flush()


# ---------------------------------------------------------------------------
# GPU memory management
# ---------------------------------------------------------------------------

def free_gpu_memory() -> None:
    """Release cached GPU tensors + run Python GC + trim glibc heap.

    Called at the end of every episode so transient allocations (GraspGen
    diffusion candidates, SAM3 / RoboRefer intermediates, Blender handoff
    buffers) don't accumulate and trigger CUDA OOM.

    Persistent model weights (Molmo / RoboRefer / SAM3 / GraspGen) are
    NOT freed — they stay resident for the next episode.

    The malloc_trim(0) call is critical: glibc keeps freed memory in
    per-thread arenas and never returns it to the OS unless explicitly
    asked. Over hundreds of episodes this grows RSS by ~400-500 MB/episode.
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass

    # Force glibc to return free heap pages to the OS.
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

def safe_close_demo(demo_obj) -> None:
    """Close a RobotEnvUI without letting teardown errors propagate."""
    if demo_obj is None:
        return
    try:
        demo_obj.close()
    except Exception as e:
        print(f"[WARN] demo.close() raised during cleanup: {e}")


def reencode_video(input_path: str, output_fps: int = VIDEO_OUTPUT_FPS):
    """Re-encode a video at a lower framerate for easier viewing.

    PyBullet's startStateLogging records at simulation step rate which
    produces very fast videos. This re-encodes at `output_fps`.
    The original file is replaced with the re-encoded version.
    """
    if not os.path.isfile(input_path):
        return
    tmp_path = input_path + ".tmp.mp4"
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, output_fps, (width, height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
        writer.release()
        os.replace(tmp_path, input_path)
    except Exception as e:
        print(f"[WARN] Failed to re-encode video {input_path}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Failure reason parsing
# ---------------------------------------------------------------------------

def parse_fail_reason(reason: str) -> str:
    """Categorise a failure reason string into a general bucket."""
    if reason is None:
        return "unknown_failure"
    r = str(reason).lower()
    if "budget exhausted" in r:
        return "action_budget_exhausted"
    if "target object" in r and "not found" in r:
        return "target_not_found"
    if "accidentally removed" in r:
        return "target_accidentally_removed"
    if "maximum vlm steps" in r:
        return "max_vlm_steps_exceeded"
    if "no answer" in r or "without a final answer" in r:
        return "no_answer_provided"
    if "parse" in r or "error" in r:
        return "vlm_parse_error"
    return "unknown_failure"
