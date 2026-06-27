# User Guide

## Overview

TERRA is a foundation model for spatial transcriptomics based on a
**Joint-Embedding Predictive Architecture (JEPA)**. Each cell is tokenized
together with its spatial neighbors into a sequence of gene tokens; parts of the
sequence are masked, and the model is trained to predict the *latent*
representations of the masked cell and neighborhood tokens rather than raw
counts. This yields embeddings that summarize both a cell's own expression and
its surrounding tissue microenvironment.

TERRA is pretrained on **HST-Corpus-112M**, a corpus of more than 100 million
cells at single-cell resolution spanning human spatial-transcriptomics datasets.
The pretrained embeddings
transfer to a range of downstream tasks without task-specific retraining.

## The inference pipeline

The user-facing API is exposed at the top level of `terra` (its implementation
lives in `terra.inference`). The typical workflow has three stages, exposed both
as a single convenience function and as individual steps:

1. **Harmonize** — map gene names to Ensembl IDs and apply quality control so the
   input matches the gene vocabulary the model was trained on
   (`harmonize_adata`).
2. **Tokenize** — build the per-cell neighborhood token sequences against a
   trained model's tokenizer (`tokenize_adata`).
3. **Embed** — run the model to retrieve cell- and neighborhood-level embeddings
   (`embed_dataset`).

The convenience wrapper `harmonize_tokenize_embed_pipeline` runs all three:

```python
from terra import download_pretrained, harmonize_tokenize_embed_pipeline

model_dir = download_pretrained("Lotfollahi-lab/TERRA-96M")

adata = harmonize_tokenize_embed_pipeline(
    adata=adata,                       # raw counts in adata.X
    sample_key="sample",
    batch_key="batch",
    model_folder_path=model_dir,
    cache_directory_path="./terra_cache",
)
```

The resulting cell- and neighborhood-level embeddings are stored in
`adata.obsm`. See the {doc}`tutorials` for the step-by-step version and downstream
analyses, and the {doc}`api` for the full reference.

### Running the steps individually

The wrapper is equivalent to calling the three steps yourself, which is useful
when you want to inspect or cache an intermediate result — the harmonized
`AnnData` or the tokenized `dataset`. The gene-reference files the harmonizer
needs live inside the downloaded bundle:

```python
import os

from terra import (download_pretrained, harmonize_adata, tokenize_adata,
                   embed_dataset)

model_dir = download_pretrained("Lotfollahi-lab/TERRA-96M")

# 1. Harmonize: map gene symbols -> Ensembl IDs and apply QC, using the bundle's
#    gene-reference files (min_*_per_* = 0 mirrors the pipeline default).
adata = harmonize_adata(
    adata,                              # raw counts in adata.X
    gene_mapping_dict_file_path=os.path.join(model_dir, "ensembl_dictionary.pkl"),
    gene_occurrence_count_file_path=os.path.join(model_dir, "gene_count_dictionary.pkl"),
    min_genes_per_cell=0,
    min_cells_per_gene=0,
)

# 2. Tokenize: build the per-cell neighborhood token sequences.
dataset = tokenize_adata(
    adata=adata,
    model_folder_path=model_dir,
    cache_directory_path="./terra_cache",
)

# 3. Embed: cell- and neighborhood-level embeddings.
embeddings = embed_dataset(dataset=dataset, model_folder_path=model_dir)
for key, values in embeddings.items():   # cell_emb, neighborhood_emb, spatial_cell_emb
    adata.obsm[key] = values
```

The tokenized `dataset` is also what the downstream functions
(`get_average_gene_embed`, `get_spatial_score`, `perturb_dataset`, …) consume, so
you can persist it with `dataset.save_to_disk(...)` and reload it later. For
multi-sample data, tokenize each sample separately and concatenate the resulting
datasets — this is exactly what the pipeline's `sample_key` does, and it keeps
spatial neighborhoods from crossing sample boundaries; harmonizing and tokenizing
the whole object at once treats it as a single sample.

:::{note}
TERRA reports progress through the standard `logging` module rather than
printing to stdout. To see progress messages (for example in a notebook), enable
logging once:

```python
import logging
logging.basicConfig(level="INFO")
```

The command-line entry points (`terra-hub`, `terra.training`, `terra.inference`)
configure this for you automatically.
:::

## Pretrained models

Pretrained TERRA models are distributed on the
[Hugging Face Hub](https://huggingface.co/Lotfollahi-lab). Each model is a
self-contained *bundle* — the checkpoint, model config, token dictionary, and the
gene-reference files needed to reproduce the model's training-time harmonization
(`ensembl_dictionary.pkl`, `gene_count_dictionary.pkl`).

| Model | Training data |
| --- | --- |
| `TERRA-96M` | A 96M-cell subset of HST-Corpus-112M; the remaining cells are held out for benchmarking and downstream analyses. |
| `TERRA-112M` | The full HST-Corpus-112M. |

Download a bundle with `download_pretrained`; the returned folder is what you pass
as `model_folder_path`:

```python
from terra import download_pretrained

model_dir = download_pretrained("Lotfollahi-lab/TERRA-96M")          # latest
model_dir = download_pretrained("Lotfollahi-lab/TERRA-96M", revision="v1.0")  # pinned
```

Because the gene-reference files are part of the bundle, harmonization at
inference time reproduces the tokenization the model was trained with — no
external file paths are required.

## Citation

If you use TERRA in your research, please cite the manuscript (in preparation). A
BibTeX entry and DOI will be added on publication.
