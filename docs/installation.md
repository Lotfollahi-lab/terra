# Installation

TERRA is published on PyPI as `terra-st` (the bare `terra` name was already taken;
the import name is still `terra`).

```shell
pip install terra-st
```

For a development install from a clone of the [repository](https://github.com/Lotfollahi-lab/terra):

```shell
git clone https://github.com/Lotfollahi-lab/terra.git
cd terra
pip install -e ".[dev,test,doc]"
```

## PyTorch / GPU note

TERRA depends on [PyTorch](https://pytorch.org). A plain `pip install terra-st`
pulls the **default** PyTorch wheel from PyPI, which on Linux is a CUDA build —
and that build must match your machine's NVIDIA driver. If the bundled CUDA is
newer than your driver, CUDA fails to initialize at runtime with an error like:

```text
RuntimeError: The NVIDIA driver on your system is too old (found version 12040).
```

PyPI cannot host the CUDA-specific PyTorch wheels (they live on
`download.pytorch.org`), so the CUDA build is an **install-time** choice. Install
the PyTorch build for your hardware **first**, then install TERRA:

```shell
# GPU — pick the CUDA version that matches your driver (see below):
pip install torch --index-url https://download.pytorch.org/whl/cu124
# ...or CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install terra-st
```

Find your driver's maximum supported CUDA version in the top-right of
`nvidia-smi` ("CUDA Version"); the installed `torch.version.cuda` must be **≤**
that value. Verify the install with:

```shell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Optional extras

TERRA ships several optional dependency groups:

| Extra | Install | Purpose |
| --- | --- | --- |
| `hub` | `pip install "terra-st[hub]"` | Publish/download model bundles on the Hugging Face Hub (`terra-hub`). |
| `eval` | `pip install "terra-st[eval]"` | Evaluation utilities (CellPhoneDB, Omnipath). |
| `doc` | `pip install "terra-st[doc]"` | Build the documentation. |
| `test` | `pip install "terra-st[test]"` | Run the test suite. |

## Reproducible environment

For the exact, fully-pinned environment TERRA is developed and tested against, use
the committed lockfile with [uv](https://docs.astral.sh/uv/):

```shell
uv sync
```
