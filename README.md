# TERRA

[![PyPI](https://img.shields.io/pypi/v/terra-st.svg)](https://pypi.org/project/terra-st/)
[![Documentation](https://readthedocs.org/projects/terra/badge/?version=latest)](https://terra.readthedocs.io/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

**TERRA** is a foundation model for spatial transcriptomics. It uses a
Joint-Embedding Predictive Architecture (JEPA): cells are tokenized together with
their spatial neighbours, parts of the input are masked, and the model learns by
predicting the *latent* representations of the masked cell and neighbourhood
tokens. The resulting embeddings capture both a cell's own expression and its
tissue microenvironment.

Pretrained on **HST-Corpus-112M** (>100M cells spanning human spatial-transcriptomics
datasets), TERRA produces cell- and neighbourhood-level embeddings that transfer to
downstream tasks such as niche and cell-type identification, batch-integrated
atlasing, spatial gene-pair scoring, and in-silico perturbation — without
task-specific retraining.

- 📖 **Documentation:** <https://terra.readthedocs.io>
- 🤗 **Pretrained models:** <https://huggingface.co/Lotfollahi-lab> (`TERRA-96M`, `TERRA-112M`)
- 📓 **Tutorial:** [end-to-end walkthrough](https://terra.readthedocs.io/en/latest/tutorials.html)

## Key features

- **Spatially-aware embeddings** — cell and neighbourhood representations learned in latent space via JEPA.
- **Pretrained and ready to use** — download a model from the Hugging Face Hub and embed your own `AnnData` in a few lines.
- **Self-contained model bundles** — each release ships the checkpoint, tokenizer, and gene-reference files needed to reproduce its training-time harmonization.
- **Downstream analyses** — niche/cell-type clustering, gene-pair spatial scoring, EMD-based spatial structure, and perturbation.

## Installation

TERRA is published on PyPI as `terra-st` (the bare `terra` name was already
taken; the import name is still `terra`):

```shell
pip install terra-st
```

For a development install from a clone of this repository:

```shell
pip install -e ".[dev,test,doc]"
```

### PyTorch / GPU note

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

## Quickstart

Download a pretrained model and embed your own spatial `AnnData` with the
end-to-end pipeline. Each downloaded bundle contains the gene-reference files
needed for harmonization, so no external paths are required:

```python
import anndata as ad
from terra import download_pretrained, harmonize_tokenize_embed_pipeline

adata = ad.read_h5ad("my_spatial_data.h5ad")   # raw counts in adata.X

model_dir = download_pretrained("Lotfollahi-lab/TERRA-96M")

adata = harmonize_tokenize_embed_pipeline(
    adata=adata,
    sample_key="sample",            # column in adata.obs identifying samples
    batch_key="batch",              # column to store the batch identifier
    model_folder_path=model_dir,
    cache_directory_path="./terra_cache",
)

# Cell- and neighbourhood-level embeddings are now in adata.obsm.
```

See the [documentation](https://terra.readthedocs.io) for the step-by-step
pipeline, downstream analyses (niche identification, gene-pair scoring,
perturbation), and the full [tutorial](https://terra.readthedocs.io/en/latest/tutorials.html).

## Citation

If you use TERRA in your research, please cite the manuscript (in preparation).
A BibTeX entry and DOI will be added here on publication.

## License

The TERRA **code** is released under the [BSD 3-Clause License](LICENSE).
Pretrained **model weights** distributed on the Hugging Face Hub are released
under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
