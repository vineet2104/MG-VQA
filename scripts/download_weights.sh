#!/usr/bin/env bash
# Download large model weights that exceed GitHub LFS limits (>2GB).
# Run once after cloning:
#   bash scripts/download_weights.sh
#
# Requires: pip install huggingface_hub

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SAM3_DIR="$REPO_ROOT/third_party/sam3"

echo "=== Downloading SAM3 weights from facebook/sam3 ==="
echo "Target: $SAM3_DIR"
echo ""

# model.safetensors (3.3 GB)
if [ -f "$SAM3_DIR/model.safetensors" ]; then
    echo "✓ model.safetensors already exists ($(du -h "$SAM3_DIR/model.safetensors" | cut -f1))"
else
    echo "Downloading model.safetensors (3.3 GB)..."
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='facebook/sam3',
    filename='model.safetensors',
    local_dir='$SAM3_DIR',
    local_dir_use_symlinks=False,
)
print('✓ model.safetensors downloaded')
"
fi

# sam3.pt (3.3 GB)
if [ -f "$SAM3_DIR/sam3.pt" ]; then
    echo "✓ sam3.pt already exists ($(du -h "$SAM3_DIR/sam3.pt" | cut -f1))"
else
    echo "Downloading sam3.pt (3.3 GB)..."
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='facebook/sam3',
    filename='sam3.pt',
    local_dir='$SAM3_DIR',
    local_dir_use_symlinks=False,
)
print('✓ sam3.pt downloaded')
"
fi

echo ""

# ── CCO wood textures for random table surfaces ──────────────
TEXTURE_DIR="$HOME/.cache/robo_blender/table_textures/cco_textures"
# NOTE: needs `|| true` — under `set -euo pipefail` an unmatched glob makes
# `ls` exit non-zero, which aborts the whole script on a fresh machine.
WOOD_COUNT=$( { ls -d "$TEXTURE_DIR"/Wood* "$TEXTURE_DIR"/PaintedWood* 2>/dev/null || true; } | wc -l)

echo "=== Downloading CCO wood textures for table rendering ==="
echo "Target: $TEXTURE_DIR"
echo ""

if [ "$WOOD_COUNT" -ge 100 ]; then
    echo "✓ Wood textures already present ($WOOD_COUNT sets)"
else
    echo "Downloading cco_textures.zip from HuggingFace (~2.5 GB)..."
    echo "(Only wood textures will be extracted)"
    ZIPFILE=$(python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='TontonTremblay/synthetic-scene-content',
    filename='cco_textures.zip',
    repo_type='dataset',
)
print(path)
")
    echo "Downloaded: $ZIPFILE"
    echo "Extracting wood textures..."
    mkdir -p "$TEXTURE_DIR"
    # NOTE: `|| true` — unzip exits 11 when one of the glob patterns matches
    # nothing (e.g. WoodFloor*), which aborts the script under `set -e`.
    unzip -qo "$ZIPFILE" "cco_textures/Wood*" "cco_textures/PaintedWood*" "cco_textures/WoodFloor*" -d "$(dirname "$TEXTURE_DIR")" || true
    WOOD_COUNT=$( { ls -d "$TEXTURE_DIR"/Wood* "$TEXTURE_DIR"/PaintedWood* 2>/dev/null || true; } | wc -l)
    echo "✓ Extracted $WOOD_COUNT wood texture sets"
fi

echo ""

# ── GraspGen 6-DOF grasp model checkpoints ───────────────────
GRASPGEN_DIR="$REPO_ROOT/checkpoints/graspgen"

echo "=== Downloading GraspGen checkpoints (Robotiq 2F-140) ==="
echo "Target: $GRASPGEN_DIR"
echo ""

mkdir -p "$GRASPGEN_DIR"

GRASPGEN_FILES=(
    "checkpoints/graspgen_robotiq_2f_140.yml"
    "checkpoints/graspgen_robotiq_2f_140_gen.pth"
    "checkpoints/graspgen_robotiq_2f_140_dis.pth"
)

ALL_PRESENT=true
for f in "${GRASPGEN_FILES[@]}"; do
    BASENAME="$(basename "$f")"
    if [ ! -f "$GRASPGEN_DIR/$BASENAME" ]; then
        ALL_PRESENT=false
        break
    fi
done

if $ALL_PRESENT; then
    echo "✓ GraspGen checkpoints already present:"
    du -sh "$GRASPGEN_DIR"/graspgen_robotiq_2f_140* 2>/dev/null || true
else
    echo "Downloading from huggingface.co/adithyamurali/GraspGenModels (~490 MB)..."
    python -c "
from huggingface_hub import hf_hub_download
files = [
    'checkpoints/graspgen_robotiq_2f_140.yml',
    'checkpoints/graspgen_robotiq_2f_140_gen.pth',
    'checkpoints/graspgen_robotiq_2f_140_dis.pth',
]
for f in files:
    import os
    basename = os.path.basename(f)
    print(f'  Downloading {basename}...')
    hf_hub_download(
        repo_id='adithyamurali/GraspGenModels',
        filename=f,
        local_dir='$GRASPGEN_DIR/_tmp',
        local_dir_use_symlinks=False,
    )
    # Move from nested checkpoints/ subdir to flat target
    src = os.path.join('$GRASPGEN_DIR/_tmp', f)
    dst = os.path.join('$GRASPGEN_DIR', basename)
    os.rename(src, dst)
    print(f'  ✓ {basename}')
import shutil
shutil.rmtree('$GRASPGEN_DIR/_tmp', ignore_errors=True)
"
    echo "✓ GraspGen checkpoints downloaded"
fi

echo ""
echo "=== Done ==="
du -sh "$SAM3_DIR/model.safetensors" "$SAM3_DIR/sam3.pt" 2>/dev/null || true
echo "Wood textures: $WOOD_COUNT sets in $TEXTURE_DIR"
du -sh "$GRASPGEN_DIR"/graspgen_robotiq_2f_140* 2>/dev/null || true
