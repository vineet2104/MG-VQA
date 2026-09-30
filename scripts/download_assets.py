#!/usr/bin/env python3
"""Download object assets from HuggingFace into owg_robot/assets/.

Pulls vineet-datasets/agentic-mqa-objects into the working repo.
Idempotent — re-running only fetches files that are missing or have
changed (the underlying snapshot_download() uses content-addressed
caching).

Required for both:
  - Benchmark eval (uses object_annotations_eval.json)
  - SFT data generation (uses object_annotations_train.json)

Usage:
    python scripts/download_assets.py
    python scripts/download_assets.py --verify     # only verify, don't fetch
    python scripts/download_assets.py --force      # re-download even if present
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "vineet-datasets/agentic-mqa-objects"
REPO_TYPE = "dataset"
ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "owg_robot" / "assets"

# The two annotation files in the repo tell us which obj_folders we
# actually need to be present after the download.
ANN_EVAL = ASSETS / "object_annotations_eval.json"
ANN_TRAIN = ASSETS / "object_annotations_train.json"

# CC0 wood texture cache used by Blender's table material.
# Path matches the hardcoded _TABLE_TEXTURE_DIR in robo_blender/renderer.py.
CCO_CACHE = Path.home() / ".cache" / "robo_blender" / "table_textures" / "cco_textures"


def required_obj_paths(include_train=False) -> dict[str, set[str]]:
    """Return {parent_dirname: {obj_folder, ...}} from both annotation files."""
    needed = {"unified_objects": set(), "dtc_objects": set()}
    for fn in ((ANN_EVAL, ANN_TRAIN) if include_train else (ANN_EVAL,)):
        if not fn.is_file():
            print(f"WARN: missing annotations {fn} — skipping verification "
                  f"for that pool.", file=sys.stderr)
            continue
        for entry in json.loads(fn.read_text()):
            src = entry.get("source", "")
            obj = entry["obj_name"]
            if src == "unified":
                needed["unified_objects"].add(obj)
            elif src == "dtc":
                needed["dtc_objects"].add(obj)
    return needed


def verify(needed: dict[str, set[str]]) -> tuple[int, int]:
    """Return (n_present, n_missing) — checks model.urdf for each ref."""
    present = missing = 0
    for parent, objs in needed.items():
        pdir = ASSETS / parent
        for obj in objs:
            if (pdir / obj / "model.urdf").is_file():
                present += 1
            else:
                missing += 1
    return present, missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true",
                   help="Only check assets are present; do not download.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if all files appear present.")
    p.add_argument("--token", type=str, default=None,
                   help="HuggingFace token if the dataset is private "
                        "(else relies on `huggingface-cli login` cache).")
    p.add_argument("--skip_textures", action="store_true",
                   help="Skip the CC0 wood texture download. Without these, "
                        "the Blender renderer's random table material falls "
                        "back to the plain URDF color (no crash). Default: off.")
    p.add_argument("--include-train", action="store_true",
                   help="Also fetch the legacy training asset pool (not needed for Bench).")
    args = p.parse_args()

    needed = required_obj_paths(args.include_train)
    n_need = sum(len(s) for s in needed.values())
    print(f"References across eval+train annotations: "
          f"{len(needed['unified_objects'])} unified + "
          f"{len(needed['dtc_objects'])} dtc = {n_need}")

    present, missing = verify(needed)
    print(f"Currently on disk: {present} present, {missing} missing")

    if missing == 0 and not args.force:
        if args.verify:
            print("✓ all required assets present.")
            return
        print("All meshes present; checking/downloading textures unless skipped.")

    if args.verify:
        print(f"⚠ {missing} required files missing; re-run without --verify.",
              file=sys.stderr)
        sys.exit(1)

    print(f"\nDownloading {REPO_ID} → {ASSETS} (object meshes) ...")
    print("  (resumable; safe to Ctrl-C and re-run)\n")

    # Pull just the object-mesh subfolders into owg_robot/assets/.
    # The cco_textures/ subfolder lives inside the same HF dataset but
    # belongs at a separate runtime path (~/.cache/...), so we fetch it
    # with a second snapshot_download() call below.
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(ASSETS),
        token=args.token,
        allow_patterns=[f"{parent}/{obj}/**"
                        for parent, names in needed.items() for obj in sorted(names)],
        ignore_patterns=[".gitattributes", "README.md"],
    )

    # CC0 wood textures: only needed for the Blender renderer's random
    # table material. Without them, table renders fall back to the URDF
    # default color (no crash). Download is small (~737 MB) so we do it
    # by default; pass --skip_textures to disable.
    if not args.skip_textures:
        print(f"\nDownloading {REPO_ID}:cco_textures/ → {CCO_CACHE} ...")
        CCO_CACHE.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=str(CCO_CACHE),
            token=args.token,
            allow_patterns=["cco_textures/**"],
        )
        # snapshot_download mirrors the repo path, so files land at
        # CCO_CACHE/cco_textures/<TexName>/...  Move them up one level
        # to match what robo_blender/renderer.py expects.
        nested = CCO_CACHE / "cco_textures"
        if nested.is_dir():
            for entry in nested.iterdir():
                target = CCO_CACHE / entry.name
                if not target.exists():
                    entry.rename(target)
            try:
                nested.rmdir()
            except OSError:
                pass

    present, missing = verify(needed)
    print(f"\nFinal: {present} present, {missing} missing")
    if missing:
        print(f"⚠ {missing} files still missing after download; "
              f"check HF dataset {REPO_ID}.", file=sys.stderr)
        sys.exit(1)
    print("✓ all required assets present.")


if __name__ == "__main__":
    main()
