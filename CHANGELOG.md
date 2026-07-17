# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.1.8] - 2026-07-17

### Fixed

-   Tokenization no longer fails under pandas 3.0; `CellGraphTokenizer` and
    `CellNeighborhoodTokenizer` now index gene metadata positionally.

### Documentation

-   Expanded the in-silico perturbation and spatial-mapping tutorials, including
    a Colab / fresh-environment install cell.
-   Documented the optional `perturb` extra and RAPIDS GPU acceleration in the
    installation guide.

## [0.1.7] - 2026-07-10

### Added

-   In-silico perturbation scoring in `terra.inference`: `infer_token_distance`
    computes per-cell Sinkhorn/energy/MMD distances between the unperturbed and
    perturbed token-embedding clouds, and `summarize_{w1,w2,energy,mmd,cosine_sim}_by_label`
    summarize a perturbation's effect across cell populations (niches or cell
    types). Requires the new optional `perturb` extra (GeomLoss).
-   `perturb_dataset(..., return_perturbation_flags=True)` attaches one boolean
    column per perturbation row marking the cells each edit affects; off by
    default, with no schema or performance change when unused.
-   Tutorials for in-silico perturbation scoring and for spatial mapping of
    perturbation effects.

### Changed

-   Constrain `datasets<5`; the 5.x column-handling change breaks
    `perturb_dataset`'s flag columns.

### Documentation

-   API reference: `Returns` sections render as a consistent bullet list, the
    duplicate return type is removed, and docstrings are corrected (parameter
    coverage, accurate return descriptions, `nproc`, summaries).
-   Downstream-analysis tutorial and prose polish.

## [0.1.6] - 2026-06-29

### Fixed

-   Zero-shot in-silico perturbation (`perturb_dataset`) no longer hangs and runs
    substantially faster, especially on larger datasets and many-core machines.

### Changed

-   `perturb_dataset` now maps in a single process by default (`nproc=1`).

## [0.1.5] - 2026-06-28

### Added

-   `terra.__version__` exposes the installed package version.

### Changed

-   `perturb_dataset` matches a specific `perturbed_cell_id` on each cell's own
    `cell_id`, so perturbing specific cells with a `cell` target no longer needs
    `add_neigh_cell_ids`. Only `neighborhood`-target perturbations on specific
    cells require it (to know each cell's neighbors).

### Fixed

-   `perturb_dataset` no longer stalls on datasets that carry a neighborhood
    `cell_ids` column. The perturbation map no longer adds per-row flag columns
    (changing the output schema forced `datasets` to re-encode every column,
    including the large nested `cell_ids`); `return_only_perturbed_cells` now
    selects the affected cells computed directly from the perturbation table.
-   `get_spatial_score` now honors its `compute_cosine_with_list` argument
    (it was previously overwritten and ignored).

### Documentation

-   Expanded the tutorials (single- and multi-sample zero-shot quickstarts and a
    downstream-analysis walkthrough covering gene programs, spatial gene-pair
    scoring, EMD spatial structure, and in-silico perturbation) and user guide,
    and fixed numerous API-reference rendering issues.

## [0.1.4] - 2026-06-27

### Fixed

-   Perturbing specific cells by ID now works on multi-sample data: the pipeline
    forwards `add_neigh_cell_ids` through its multi-sample branch, and the
    neighborhood `cell_ids` column is kept out of the torch format so it
    survives tokenization.

## [0.1.3] - 2026-06-27

### Fixed

-   `perturb_dataset` no longer hangs on torch-formatted datasets. It now maps
    over the unformatted dataset (formatting the large nested token columns
    inside `map` could stall indefinitely) and restores the format afterwards.

### Changed

-   `perturb_dataset(..., return_only_perturbed_cells=True)` now subsets the
    perturbed cells with an index-based `select` instead of a full-table
    `filter`, avoiding a costly scan and rewrite of every tokenized row.

## [0.1.2] - 2026-06-27

### Fixed

-   `read_xenium_10x` now aligns cell centroids correctly for older Xenium
    panels with an integer `cell_id`, which previously produced all-NaN spatial
    coordinates.
-   `get_spatial_score` and `get_emd_distance` no longer raise
    `NameError: gene_embed_dataset` — the helper they depend on is now imported.
-   `perturb_dataset` with `perturbed_cell_id="all"` now applies `"all"`-gene
    knockout/fold-change perturbations correctly (the all-cells fast path
    referenced an unbound variable for whole-panel perturbations).

## [0.1.1] - 2026-06-26

### Added

-   Optional `local_dir` argument to `download_pretrained` to download a model
    bundle into a chosen folder instead of the Hugging Face cache.

### Fixed

-   Single-sample zero-shot inference (`harmonize_tokenize_embed_pipeline` with
    `sample_key=None`) no longer fails with `KeyError: 'batch'`; the whole
    AnnData is now treated as one batch and any existing batch column is kept.

## [0.1.0] - 2026-06-26

First public release of **TERRA**, a JEPA-based foundation model for spatial
transcriptomics, pretrained on HST-Corpus-112M (>100 million cells at
single-cell resolution).

### Added

-   Pretrained models (`TERRA-96M`, `TERRA-112M`) distributed as self-contained
    bundles on the Hugging Face Hub, with download and publishing utilities
    (`download_pretrained`, `terra-hub`).
-   Zero-shot inference pipeline producing cell- and neighborhood-level
    embeddings for single- and multi-sample spatial data.
-   Downstream analyses: gene-level embeddings, spatial gene-pair scoring,
    EMD-based spatial structure, and in-silico perturbation.
-   Finetuning of the pretrained encoder with LoRA/PEFT.
-   Documentation, tutorials, and API reference.

[0.1.8]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.8
[0.1.7]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.7
[0.1.6]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.6
[0.1.5]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.5
[0.1.4]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.4
[0.1.3]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.3
[0.1.2]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.2
[0.1.1]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.1
[0.1.0]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.0
