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

## Hugging Face Hub

```{eval-rst}
.. currentmodule:: terra

.. autosummary::
    :toctree: generated

    download_pretrained
    push_model_to_hub
```
