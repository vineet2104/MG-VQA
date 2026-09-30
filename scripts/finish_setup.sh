#!/usr/bin/env bash
# Install GPU runtime into the CURRENT conda environment. Never uses sudo,
# removes CUDA packages, changes drivers, or silently tolerates build failures.
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA=auto
DRY=0
SKIP_POINTNET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) CUDA="${2:?expected 11.8, 12.1 or 12.8}"; shift 2;;
    --dry-run) DRY=1; shift;;
    --skip-pointnet) SKIP_POINTNET=1; shift;;
    -h|--help)
      printf '%s\n' 'Usage: bash scripts/finish_setup.sh [--cuda auto|11.8|12.1|12.8] [--dry-run] [--skip-pointnet]' \
        'Activate your conda environment first. Install a matching CUDA toolkit + compatible C++ compiler separately.' \
        'Respect CUDA_HOME, CC, CXX, TORCH_CUDA_ARCH_LIST, MAX_JOBS. --skip-pointnet is for direct/perception only.'
      exit 0;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
if [[ -z "${CONDA_PREFIX:-}" ]]; then echo 'Activate a conda environment first.' >&2; exit 2; fi
if [[ "$CUDA" == auto ]]; then
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -V | tail -1)
  if [[ "${CAP%%.*}" -ge 10 ]]; then CUDA=12.8; else CUDA=12.1; fi
fi
case "$CUDA" in
  11.8) TORCH=2.3.1; VISION=0.18.1; TAG=cu118;;
  12.1) TORCH=2.3.1; VISION=0.18.1; TAG=cu121;;
  12.8) TORCH=2.7.0; VISION=0.22.0; TAG=cu128;;
  *) echo "Unsupported CUDA profile: $CUDA" >&2; exit 2;;
esac
run() { printf '+ '; printf '%q ' "$@"; printf '\n'; if [[ "$DRY" == 0 ]]; then "$@"; fi; }
export MAX_JOBS="${MAX_JOBS:-4}"
if [[ "$SKIP_POINTNET" == 0 && "$DRY" == 0 ]]; then
  NVCC="${CUDA_HOME:+$CUDA_HOME/bin/}nvcc"
  command -v "$NVCC" >/dev/null || { echo "Missing nvcc. Install CUDA $CUDA or set CUDA_HOME; see INSTALL.md." >&2; exit 1; }
  "$NVCC" --version | grep -q "release $CUDA," || { echo "nvcc must match selected CUDA $CUDA." >&2; exit 1; }
fi
run python -m pip install "torch==$TORCH" "torchvision==$VISION" --index-url "https://download.pytorch.org/whl/$TAG"
run python -m pip install -r requirements-runtime.txt
run python -m pip install --only-binary=:all: --no-deps torch-scatter==2.1.2 -f "https://data.pyg.org/whl/torch-$TORCH+$TAG.html"
if [[ "$SKIP_POINTNET" == 0 ]]; then
  # Upstream hardcodes obsolete CUDA architectures; patch only the temporary build.
  if [[ "$DRY" == 1 ]]; then
    echo '+ clone Pointnet2_PyTorch, respect TORCH_CUDA_ARCH_LIST, build pointnet2_ops_lib'
  else
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(python -c 'import torch; print(";".join(sorted({"%d.%d" % torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})))')}"
    [[ -n "$TORCH_CUDA_ARCH_LIST" ]] || { echo 'Set TORCH_CUDA_ARCH_LIST when building without a GPU.' >&2; exit 1; }
    BUILD=$(mktemp -d)
    trap 'rm -rf "$BUILD"' EXIT
    git clone --depth 1 https://github.com/erikwijmans/Pointnet2_PyTorch.git "$BUILD/p2"
    python - "$BUILD/p2/pointnet2_ops_lib/setup.py" <<'PY'
import pathlib, re, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
s, n = re.subn(r'os.environ\["TORCH_CUDA_ARCH_LIST"\]\s*=.*',
               'os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")', s)
if n != 1:
    raise RuntimeError("Upstream setup.py changed; review architecture patch")
p.write_text(s)
PY
    run python -m pip install --no-build-isolation "$BUILD/p2/pointnet2_ops_lib"
  fi
fi
run python -m pip check
echo 'GPU dependency install complete. Download assets/weights, set BLENDER, then run the preflight.'
