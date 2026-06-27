# User Guide

## Overview

TERRA is a self-supervised foundation model for spatial transcriptomics based on a
**Joint-Embedding Predictive Architecture (JEPA)**. A context-aware tokenization
serializes each cell together with its spatial neighbors into a sequence of **gene
tokens** — cells ordered by distance from the index cell, genes ranked by abundance
within each cell — which preserves gene identity and works across heterogeneous
gene panels. During pretraining some of these tokens are masked, and the model
predicts their *latent representations* rather than reconstructing raw counts,
inferring the masked molecular and spatial context of neighboring cells. Averaging
the resulting gene-token embeddings then yields representations at three scales:
genes, cells, and neighborhoods.

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

## In-silico perturbation

`perturb_dataset` edits the gene tokens of a tokenized `dataset`; re-embedding the
result with `embed_dataset` shows how the perturbation moves cells in latent
space. Each edit is one row of a small `perturb_df`:

| Column | Meaning |
| --- | --- |
| `perturbed_cell_id` | the cell to edit, or `"all"` for every cell |
| `perturbed_ensembl_id` | the gene to edit (Ensembl ID), or `"all"` for the whole panel |
| `perturbation_target` | `"cell"` or `"neighborhood"` (see below) |
| `perturbation_type` | `"knockout"` or `"foldchange"` (see below) |
| `foldchange` | the multiplier for `foldchange` rows (use `np.nan` for knockout) |

**Perturbation type** — *what* happens to the targeted expression:

- `knockout` sets it to zero (simulates removing the gene).
- `foldchange` multiplies it by `foldchange` (e.g. `0.5` to halve, `2.0` to double).

**Perturbation target** — *where* the edit is applied. Each cell is tokenized
together with its spatial neighbors, so a sequence has two parts:

- `cell` edits the cell's own gene tokens (its intrinsic expression).
- `neighborhood` edits the tokens contributed by the cell's spatial neighbors
  (its microenvironment), leaving the cell itself untouched.

**Scope** — `perturbed_cell_id` and `perturbed_ensembl_id` each accept a specific
value or `"all"` (every cell / the whole panel). Multiple rows are applied
together, so you can combine edits in a single run.

```python
import numpy as np
import pandas as pd
from terra import perturb_dataset, embed_dataset

# Halve all genes and knock out one gene, for every cell, applied to the cell
# itself. Add `perturbation_target="neighborhood"` rows to perturb the context.
perturb_df = pd.DataFrame([
    {"perturbed_cell_id": "all", "perturbed_ensembl_id": "all",
     "perturbation_target": "cell", "perturbation_type": "foldchange", "foldchange": 0.5},
    {"perturbed_cell_id": "all", "perturbed_ensembl_id": "ENSG00000136997",  # a gene from your panel
     "perturbation_target": "cell", "perturbation_type": "knockout", "foldchange": np.nan},
])

perturbed = perturb_dataset(dataset, perturb_df, model_folder_path=model_dir)
emb = embed_dataset(perturbed, model_folder_path=model_dir)
# Compare emb["cell_emb"] / emb["neighborhood_emb"] against the unperturbed run.
```

Pass `return_only_perturbed_cells=True` to get back just the edited cells. See
the {doc}`tutorials` (downstream analysis) for a worked before/after comparison.

## Citation

If you use TERRA in your research, please cite the manuscript (in preparation). A
BibTeX entry and DOI will be added on publication.
