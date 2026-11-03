# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

-   New `meta.attention_backend` config option selecting the scaled
    dot-product-attention (SDPA) backend: `auto` (default, unchanged
    behaviour — fp32 softmax), `flash` (lowest memory, largest usable batch
    size, bf16 softmax), `efficient`, `cudnn`, `math`, or `torch`. Previously
    the backend preference was hard-coded.
-   The attention implementation and backend preference are now logged when the
    model is initialized, so every training, finetuning and inference run
    records which attention path it used.
-   Inference runtime and throughput are now logged by `embed_dataset`,
    `gene_embed_dataset`, `infer` and `infer_token_distance`, annotated with the
    attention backend, to make runtime comparisons between configurations
    straightforward.
-   Opt-in fixes from the October 2026 code review. All defaults reproduce the
    previous behaviour, including the published TERRA-96M and TERRA-112M
    results:
    -   a top-level `seed` config key for pretraining (default 0);
    -   `keep_special_tokens` for `embed_dataset`, `gene_embed_dataset`, `infer`
        and `harmonize_tokenize_embed_pipeline`, and a warning when inference
        pads the metatokens of a model that was trained with them in every
        sequence (e.g. TERRA-112M);
    -   tokenizer options `stable_rank_ties`, `zero_expr_from_raw_counts`,
        `gcrd_basis_target_read_depth` / `gcrd_target_read_depth_per_gene`,
        `seurat_loess_span` and, for cell-neighborhood models,
        `aggregate_own_neighbors`; inference reads them from the model
        config's `data` section;
    -   `mask.pad_masks_to_batch_max`, which pads the JEPA context and target
        masks to the longest cell in a batch instead of trimming them to the
        shortest;
    -   `schedule_values_at_step`; training also logs the learning rate,
        weight decay and EMA momentum at its last step (with `ipe_scale` > 1
        the schedules end before their configured final values).
-   Further opt-in options from the November 2026 code review; their defaults
    also reproduce the previous behaviour:
    -   tokenizer option `pad_empty_neighbors` (`data.pad_empty_neighbors`):
        a neighbor without vocabulary genes keeps an all-padding segment, so
        every neighbor occupies a fixed segment on both ranking paths;
    -   tokenizer option `eliminate_stored_zeros`
        (`data.eliminate_stored_zeros`), which removes explicitly stored zeros
        from sparse count matrices before tokenization;
    -   `keep_special_tokens` for both finetuning modes
        (`finetune.keep_special_tokens`) and for `infer_token_distance`;
    -   the top-level `seed` config key for both finetuning modes (default 0);
    -   `random_state` for the subsampling of the `summarize_*_by_label`
        point-cloud summaries.

### Changed

-   The `hst_corpus_110m` model configs now use `attention_backend: flash`
    (and `use_flash_attention: True`, which flash requires).
-   `stable_rank_ties` (unreleased) now orders equally ranked genes by token
    ID on both ranking paths, so tokenization no longer depends on the order
    of the gene columns in the input.
-   `embed_dataset(agg_type='softmax')` applies the softmax once, over the
    aggregated genes only, so outputs of this option change. It was not used
    for published results; the default `agg_type='avg'` is unchanged.
-   User-facing warnings (padded metatokens at inference, cells without
    expressed genes, duplicated genes) are issued as `UserWarning`s and are
    shown regardless of the logging configuration.
-   The training launcher no longer forces `TORCH_DISTRIBUTED_DEBUG=DETAIL`;
    set it in the job environment to enable it.

### Fixed

-   A checkpoint that cannot be loaded now raises an error instead of
    silently leaving the model randomly initialized (training, finetuning and
    inference). For training, `state.allow_checkpoint_load_failure: true`
    restores the previous behaviour.
-   Resuming multi-GPU training from the latest checkpoint
    (`read_checkpoint: null`) now resumes every rank, not only rank 0. The
    latest checkpoint is written atomically.
-   Inference tokenizes cell-neighborhood models with the cell/neighborhood
    split recorded in their config. The tokenizer default changed in April
    2026.
-   `n_nonzero_tokens` now counts gene tokens on the dense ranking path, so
    length-grouped batching works for those tokenizations. It is unchanged on
    the sparse path, e.g. for the TERRA-96M/112M data.
-   Tokenizing with Pearson-residual ranks and shifted-log counts no longer
    fails with an `IndexError`.
-   Finetuning rebuilds the pretrained model from its saved config, as
    inference does. Supervised finetuning of TERRA-96M previously dropped the
    value-MLP biases of the pretrained model, and TERRA-112M could not be
    finetuned at all; finetuning results change accordingly. The finetuning
    data follow the model config as at inference, and the classification head
    no longer pools over the metatoken.
-   Inference follows the training config: `restrict_special_attention` is
    read from `mask` (no NaN embeddings with eager attention), `nz_spc` is
    passed to the datasets, cell-neighborhood models use the configured cell
    length, and models with `api_version` v4 or unset, or with neighbor
    truncation, can be embedded. `infer_token_distance` works with
    TERRA-112M-style models, and `infer()` works with its default arguments,
    with cell-neighborhood models and with `agg_type='weighted_avg'`. Outputs
    of the released models are unchanged.
-   `keep_special_tokens=True` checks its inputs and raises a clear error
    when the `adata.uns` metadata, a batch the model was trained on, or the
    special-token columns of a tokenized dataset are missing.
-   `harmonize_adata` accepts Ensembl gene IDs (with or without version
    suffix) as `var_names`, raises an error when no gene can be mapped and
    warns when fewer than 10% map. It no longer modifies the input AnnData,
    raises a clear error instead of hanging when neither `X` nor
    `layers['counts']` holds integer counts, and treats genes missing from the
    occurrence counts as unseen instead of raising a `KeyError`. Genes that
    occur in several columns are kept as before, with a warning; with
    `sample_key` the pipeline reports them with a clear error instead of an
    `InvalidIndexError`.
-   The tokenizer warns about cells without any expressed gene: they get empty
    cell embeddings and shift the neighbor segments of nearby cells. The
    pipeline still keeps them by default (`min_genes_per_cell=0`, reproducing
    existing results); pass `min_genes_per_cell=1` to remove them, as the user
    guide now does.
-   The read-depth, gene-corrected read-depth and cell-area normalizers return
    CSR matrices, which fixes tokenization with these normalizations under
    recent scipy and anndata releases.
-   Training: metatoken dropout (`mask.special_token_pad_ratio` > 0) and
    self-supervised finetuning of models with metatokens use the correct
    predictor layout when the metatokens are padded, and
    `restrict_special_attention` no longer produces NaN outputs for padded
    metatokens. Cycle consistency with metadata keys now changes the encoder
    input. `ipe_scale` < 1 works, gradient clipping stays on after a
    fractional warmup, and the logged gradient norms are no longer 0 during
    warmup. Mid-epoch checkpoints are written every `checkpoint_freq_iter`
    iterations, without the extra checkpoint after the first iteration of
    each epoch. Training of the released configurations is unchanged.
-   RDA depth conditioning starts as an exact no-op, as documented. RoPE draws
    one rotation per forward pass, its target encoder runs in eval mode, and a
    warning flags frequency scales that alias on micrometre coordinates.
-   The training launcher supports plain `srun` launches, and `meta.add_cls`
    is no longer silently ignored for cell-neighborhood models.
-   `harmonize_tokenize_embed_pipeline` accepts a `pathlib.Path` save path and
    no longer derives the batch label from it; `get_gene_embed` accepts
    repeated genes.

## [0.1.13] - 2026-07-28

### Fixed

-   Tokenization no longer crashes on **integer spatial coordinates** (e.g. 10x
    Visium pixel coordinates) with `RuntimeError: value cannot be converted to
    type int64_t without overflow`. The relative position coordinates are now
    always float, regardless of the input coordinate dtype.

## [0.1.12] - 2026-07-28

### Changed

-   The `notebook` extra now includes `gdown`. The tutorials that download demo
    data install `terra-st[notebook]` (or `terra-st[perturb,notebook]`) instead
    of pulling `gdown` in separately, and their setup cells no longer imply the
    install is Colab-only.

### Documentation

-   The Visium transfer-learning tutorial now runs end-to-end on public 10x
    Visium data — two serial Human Breast Cancer sections downloaded via
    `scanpy.datasets.visium_sge` — instead of a private local dataset, and states
    up front that it is a usage template that needs many more sections than the
    two-section demo to produce good results.

## [0.1.11] - 2026-07-28

### Changed

-   Declare the scientific-stack packages terra imports directly (`numpy`,
    `pandas`, `scipy`, `scanpy`, `scikit-learn`, `matplotlib`, `tqdm`,
    `requests`; and `ipython` for the `eval` extra) instead of relying on them
    being pulled in transitively.

### Documentation

-   Supervised fine-tuning tutorial: default `hidden_dim` and `lr` set to 128
    and 1e-4, with a note that they may need tuning for custom datasets.

## [0.1.10] - 2026-07-27

### Fixed

-   `finetune_self_supervised` no longer crashes at the start of training with
    `ValueError: too many values to unpack`; the training loop now unpacks the
    mask collator's 5-tuple batches, matching the rest of the training code.
-   Declare `setuptools<81` as a dependency so a clean (e.g. `uv`) environment
    can import the `squidpy`/`spatialdata` stack, which relies on the legacy
    `pkg_resources` module.

## [0.1.9] - 2026-07-27

### Added

-   Self-supervised LoRA fine-tuning in `terra.training.finetune_self_supervised`:
    `finetune_self_supervised` adapts a pretrained TERRA encoder to new data or a
    new platform by training only LoRA adapters (self-supervised, no labels), and
    `prepare_finetuned_model` merges the adapters into a self-contained,
    embeddable model bundle.

### Fixed

-   `prepare_finetuned_model` now copies the gene-reference files into the
    fine-tuned bundle, so embedding with a fine-tuned model reproduces the
    training-time gene mapping instead of falling back to a generic Ensembl one.

### Documentation

-   New fine-tuning tutorials: supervised label transfer, self-supervised
    fine-tuning on Xenium data, and transfer learning to Visium, with API
    reference for `finetune` and `filter_dataset_by_cell_ids`.

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

[0.1.12]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.12
[0.1.11]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.11
[0.1.10]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.10
[0.1.9]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.9
[0.1.8]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.8
[0.1.7]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.7
[0.1.6]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.6
[0.1.5]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.5
[0.1.4]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.4
[0.1.3]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.3
[0.1.2]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.2
[0.1.1]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.1
[0.1.0]: https://github.com/Lotfollahi-lab/terra/releases/tag/0.1.0
