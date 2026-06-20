# API

The main user-facing API lives in `app.inference`. The typical workflow is to
harmonize an `AnnData`, tokenize it against a trained model, and embed it:

```python
from app.inference import harmonize_tokenize_embed_pipeline
```

## Inference

```{eval-rst}
.. currentmodule:: app.inference

.. autosummary::
    :toctree: generated

    harmonize_tokenize_embed_pipeline
    harmonize_adata
    tokenize_adata
    embed_dataset
    infer
    get_gene_embed
    get_average_gene_embed
```
