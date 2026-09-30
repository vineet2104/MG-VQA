# Installation controls

## Prerequisites

Target: Linux x86_64 with an NVIDIA GPU and a compatible host driver. Start with
one worker. Blender EEVEE requires working GPU graphics/EGL drivers even in
headless mode; conda cannot install a host GPU driver. Blender 5.1 is the current
reference version. `BLENDER` overrides PATH; an invalid explicit path fails
rather than silently selecting a different binary.

## Conda and GPU packages

```bash
conda env create -f environment.yml -n my-mg-vqa
conda activate my-mg-vqa
bash scripts/finish_setup.sh --cuda 12.8 --dry-run
bash scripts/finish_setup.sh --cuda 12.8
```

The installer uses the **active** environment, not a named local installation.
It never uses sudo, removes existing CUDA packages, or modifies GPU drivers.
Use a fresh environment rather than updating an existing research environment.

| CUDA profile | Torch / torchvision | Intended hardware |
|---|---|---|
| 11.8 | 2.3.1 / 0.18.1 | Supported pre-Blackwell GPUs |
| 12.1 | 2.3.1 / 0.18.1 | Ampere / Ada / Hopper |
| 12.8 | 2.7.0 / 0.22.0 | Blackwell (also requires sufficiently new driver) |

`--cuda auto` selects 12.8 for compute capability >=10, otherwise 12.1.
Install the matching CUDA toolkit (including nvcc and development headers)
from your system/package manager or NVIDIA conda channel, e.g.:

```bash
conda install -c nvidia cuda-toolkit=12.8
# For a system toolkit instead:
export CUDA_HOME=/path/to/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
```

Use a CUDA-supported GCC/G++ pair; set `CC` and `CXX` if your default compiler
is too new. `MAX_JOBS=4` limits build RAM by default. `TORCH_CUDA_ARCH_LIST`
can override architecture detection (also needed for builds without a visible
GPU). Installation failures are fatal rather than being reported as success.
`--skip-pointnet` omits the agentic-only PointNet2 extension for direct/perception
experiments. The shared imports still require torch and the runtime requirements.

No cuRobo, DeepSpeed, FlashAttention or local VLM weights are required for the
supported gateway evaluation path. The GraspGen PointNet backend uses PointNet2;
its currently eager PTv3 imports also require torch-scatter and spconv, even
though PTv3 is not the configured backbone. This dependency can be reduced in a
later upstream-compatible cleanup.

## Data and credentials

- `python scripts/download_assets.py` fetches only evaluation-pool meshes plus
  table textures. It does not overwrite tracked annotation JSONs. `--verify`
  checks meshes; `--skip_textures` is for diagnostics, not matched evaluation.
- `bash scripts/download_weights.sh` fetches SAM3 and GraspGen. Accept any
  required upstream model terms and run `hf auth login` (or set `HF_TOKEN`) if
  access is gated. These weights are not redistributed in git.
- `python scripts/download_benchmark.py` downloads
  [bhatvineet/mg-vqa-bench](https://huggingface.co/datasets/bhatvineet/mg-vqa-bench)
  at revision `78c4d8adf7a5c68060bc32b340c8a746933f659c` into
  `approved-questions/mg-vqa-bench`. Use `--output` to change the destination,
  then pass that directory to `./run_mgvqa.sh --dataset PATH`.
- Set `NV_API_KEY` in `.env`. `NVIDIA_API_BASE_URL` optionally overrides the
  default `https://inference-api.nvidia.com/v1`; model entitlements are specific
  to this gateway, not guaranteed by a generic NVIDIA API key.
- `HF_HOME` controls the Hugging Face cache. Downloads and assets remain outside
  the public git tree. Do not commit `.env` or access tokens.

## Checks

```bash
python scripts/preflight.py --mode agentic
python -m pytest tests -q
python scripts/check_models.py --model gcp/google/gemini-3.6-flash
./run_mgvqa.sh --mode direct,perception,agentic --episodes 0,150,300,450 --workers 0
```

The API check and evaluation incur usage charges. Preflight checks imports,
weight presence and a CUDA kernel, not complete rendering/manipulation quality.
A successful check in an existing environment is not a fresh-install test.
