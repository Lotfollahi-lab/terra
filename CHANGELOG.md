# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.1.3] - 2026-06-27

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

[0.1.3]: https://github.com/Lotfollahi-lab/terra/releases/tag/v0.1.3
[0.1.2]: https://github.com/Lotfollahi-lab/terra/releases/tag/v0.1.2
[0.1.1]: https://github.com/Lotfollahi-lab/terra/releases/tag/v0.1.1
[0.1.0]: https://github.com/Lotfollahi-lab/terra/releases/tag/v0.1.0
