# API

The main user-facing API is exposed at the top level of the `terra` package. The
typical workflow is to harmonize an `AnnData`, tokenize it against a trained
model, and embed it:

```python
from terra import download_pretrained, harmonize_tokenize_embed_pipeline
```

## Inference

```{eval-rst}
.. currentmodule:: terra

.. autosummary::
    :toctree: generated

    harmonize_tokenize_embed_pipeline
    harmonize_adata
    tokenize_adata
    embed_dataset
    gene_embed_dataset
    infer
    get_gene_embed
    get_average_gene_embed
    perturb_dataset
    get_emd_distance
    get_spatial_score
```

## In-silico perturbation scoring

Quantify a perturbation's effect on the embeddings — per cell, or summarized
across cell populations (niches / cell types). These live in `terra.inference`
and require the optional `perturb` extra (`pip install "terra-st[perturb]"`).

```{eval-rst}
.. currentmodule:: terra.inference

.. autosummary::
    :toctree: generated

    infer_token_distance
    summarize_w1_by_label
    summarize_w2_by_label
    summarize_energy_by_label
    summarize_mmd_by_label
    summarize_cosine_sim_by_label
```

## Hugging Face Hub

```{eval-rst}
.. currentmodule:: terra

.. autosummary::
    :toctree: generated

    download_pretrained
    push_model_to_hub
```

## Data

Convenience reader used by the tutorials to load public 10x Genomics Xenium
samples.

```{eval-rst}
.. currentmodule:: terra.datasets

.. autosummary::
    :toctree: generated

    read_xenium_10x
```
