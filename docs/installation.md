# Installation

TERRA is published on PyPI as `terra-st` (the import name is `terra`) and
**requires an NVIDIA GPU**. Install in two steps — PyTorch first, so it matches
your GPU, then TERRA. We recommend [uv](https://docs.astral.sh/uv/).

## 1. Install PyTorch for your hardware

Install the [PyTorch](https://pytorch.org) build that matches your GPU driver
**before** installing TERRA, so the correct CUDA wheel is used (otherwise a plain
install pulls the default wheel, which may not match your driver). Run
`nvidia-smi` and read the "CUDA Version" in the top-right, then install the
matching CUDA build — see the
[official PyTorch install guide](https://pytorch.org/get-started/locally/) — e.g.:

```shell
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## 2. Install TERRA

```shell
uv pip install terra-st
```

Plain `pip install terra-st` works too. Verify the install (PyTorch sees your
GPU, and TERRA imports):

```shell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import terra; print(terra.__version__)"
```

The last value printed by the first command should be `True` — TERRA requires a
GPU, so `torch.cuda.is_available()` must report one. If it prints `False`,
PyTorch can't see your GPU (usually a CUDA build that doesn't match your driver
— revisit step 1).

## Development install

For a development install from a clone of the
[repository](https://github.com/Lotfollahi-lab/terra) (after step 1 above):

```shell
git clone https://github.com/Lotfollahi-lab/terra.git
cd terra
uv pip install -e ".[dev,test,doc]"
```

## Optional extras

TERRA ships several optional dependency groups:

| Extra | Install | Purpose |
| --- | --- | --- |
| `perturb` | `uv pip install "terra-st[perturb]"` | In-silico perturbation scoring (`infer_token_distance`, `summarize_*_by_label`). Installs [GeomLoss](https://www.kernel-operations.io/geomloss/) for the point-cloud (Sinkhorn/energy/MMD) distances. **Required by the perturbation and spatial-mapping tutorials.** |
| `hub` | `uv pip install "terra-st[hub]"` | Publish/download model bundles on the Hugging Face Hub (`terra-hub`). |
| `notebook` | `uv pip install "terra-st[notebook]"` | JupyterLab + ipykernel to run the tutorial notebooks. |
| `eval` | `uv pip install "terra-st[eval]"` | Evaluation utilities (CellPhoneDB, Omnipath). |
| `doc` | `uv pip install "terra-st[doc]"` | Build the documentation. |
| `test` | `uv pip install "terra-st[test]"` | Run the test suite. |

## Reproducible environment

For the exact, fully-pinned environment TERRA is developed and tested against, use
the committed lockfile with [uv](https://docs.astral.sh/uv/):

```shell
uv sync
```
