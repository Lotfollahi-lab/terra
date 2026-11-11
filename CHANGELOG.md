# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

-   New `meta.attention_backend` config option restricting the scaled
    dot-product-attention (SDPA) backends: `auto` (default; excludes
    FlashAttention), `flash` (allows it: lowest memory, largest usable batch
    size), `efficient`, `cudnn`, `math`, or `torch` (unconstrained, as
    TERRA-96M and TERRA-112M were trained: FlashAttention for unmasked
    attention). PyTorch tries the enabled backends in its own priority order.
    Inference always passes an attention mask, so it uses the memory-efficient
    kernel with every preset that allows it.
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

-   Opt-in options from the November 2026 review of the GPU, evaluation and
    Hub code; their defaults keep the released models' results:
    -   `meta.fp32_position_encoding` computes the phases of the coordinate
        position encodings (`coord`, `polar`, `polar+alibi`, `rope`) in
        float32 under bf16 autocast. By default they are computed in bf16 on
        GPU, as TERRA-96M was trained, and in float32 on CPU, so GPU and CPU
        embeddings of such models differ by more than the remaining bf16
        rounding;
    -   `finetune.use_bfloat16` for supervised finetuning (as for
        self-supervised finetuning); finetuning runs in float32 by default;
    -   `seed` for `linear_classifier` and `linear_regressor`;
    -   `terra-hub upload --model-card` to upload a model card from a file.

-   Tokenizer option `rank_tie_break_norm_factor`
    (`data.rank_tie_break_norm_factor`, with `stable_rank_ties`): genes with
    tied ranking values are ordered by a column of the norm-factor file
    before the token id, e.g. `mean` to put genes that are rarer or lower
    expressed in the corpus first, on both ranking paths (the count value no
    longer orders tied genes), so a ranking of raw counts keeps the same genes
    whatever the count normalization. Off by default; inference reads it from
    the model config.
-   `meta.zero_fully_masked_attention_rows`: an attention query row without
    any valid key (e.g. a cell whose JEPA context is only padding when no
    metatoken is used) outputs zeros with finite gradients instead of NaN, on
    every attention path and whatever the SDPA kernel; all other rows are
    unchanged. Off by default; finetuning and inference read it from the
    model config.
-   Safeguards for pretraining data (November 2026 pre-retraining review):
    -   `terra.datasets.utils.save_precomputed_split` writes a precomputed
        split together with the number of rows and the cell ids of evenly
        spaced rows of its dataset; such a split is refused when applied to
        another tokenization (e.g. after the corpus was re-tokenized and its
        row order changed). Bare lists of row numbers are still accepted,
        with a warning.
    -   `tokenize_data` records the settings that determine the tokens
        (normalizations, tie and neighbor options, normalization files) in
        `terra_tokenizer_params.json` inside the saved dataset. Training
        checks the model config against this record, raises on a mismatch
        and adds the settings the config leaves out, so that inference
        tokenizes new data as the training data were tokenized.
    -   `mask.min_context_genes`: a cell with more gene tokens than this
        keeps at least that many of them in the JEPA context (at mask ratio
        0.6 a cell with one or two gene tokens otherwise has all of them as
        targets). 0 by default (previous behaviour).
    -   Tokenizer option `special_tokens`: only these special token values
        are read (the others are stored as padding).

### Changed

-   Faster inference with the same results:
    -   `embed_dataset`, `gene_embed_dataset`, `infer` and
        `infer_token_distance` compute the cell-only context on the special
        tokens and the index cell alone (257 instead of 2817 tokens for
        TERRA-112M and TERRA-96M), and the full context without copying the
        attention mask to `[B, 1, L, L]`. Measured on an NVIDIA L40S with
        20,000 Xenium cells, embedding is 1.7x (TERRA-112M) and 1.5x
        (TERRA-96M) faster and needs 18-25% less GPU memory, and the
        embeddings are bit-identical to those of the previous version. On CPU
        they can differ by rounding (up to about 2e-7); setting
        `GeneTransformerBaseEncoder.layer_emb_shortcuts = False` reproduces
        the previous computation exactly. The neighborhood positions of the
        cell-only context, which no function uses, are now zero.
    -   Tokenization (`tokenize_adata`, `harmonize_tokenize_embed_pipeline`)
        formats the examples while the dataset is generated instead of
        rewriting the dataset with `map`, reads each cell's expression from
        the sparse matrix arrays and ranks its genes once. On the same 20,000
        cells it is 1.4x faster with 4 processes and 3x faster with 1; the
        tokenized datasets are identical.
-   The `hst_corpus_110m` model configs now use `attention_backend: flash`
    (and `use_flash_attention: True`, which flash requires).
-   `stable_rank_ties` (unreleased) now orders equally ranked genes by token
    ID on both ranking paths, so tokenization no longer depends on the order
    of the gene columns in the input. With it, the cell graph tokenizer also
    orders equidistant neighbors with a stable sort, so their segments do not
    depend on the CPU type.
-   `embed_dataset(agg_type='softmax')` applies the softmax once, over the
    aggregated genes only, so outputs of this option change. It was not used
    for published results; the default `agg_type='avg'` is unchanged.
-   User-facing warnings (padded metatokens at inference, cells without
    expressed genes, duplicated genes) are issued as `UserWarning`s and are
    shown regardless of the logging configuration.
-   The training launcher no longer forces `TORCH_DISTRIBUTED_DEBUG=DETAIL`;
    set it in the job environment to enable it.
-   The training launcher honours `WANDB_MODE` (e.g. `offline` or
    `disabled`) instead of always logging to wandb online; without it, runs
    log online as before.
-   Pretraining DataLoader workers are started with `fork` on Linux (the
    platform default elsewhere), set explicitly. This is what ran so far
    (the `spawn` start method requested in `train` never took effect after
    loading the dataset from disk), and it keeps working from Python 3.14,
    whose `forkserver` default fails with large datasets.
-   Importing TERRA no longer configures the root logger; TERRA's log records
    are shown once the application configures logging (e.g.
    `logging.basicConfig(level="INFO")`).
-   Supervised finetuning shuffles the training data in every epoch with the
    configured seed; `finetune.shuffle: false` restores the previous fixed
    order. Self-supervised finetuning samples gene masks unless
    `data.sample_gene_masks` is set to false (it failed without the key).
-   Fold changes in `perturb_dataset` multiply the counts: for shifted-log
    models such as TERRA-112M and TERRA-96M a value `v` becomes
    `log1p(expm1(v) * foldchange)` (previously the log values were
    multiplied).
-   `gene_embed_dataset(return_distance=True)` and `get_emd_distance` require
    the same cell and neighborhood genes in the same order and raise a clear
    error otherwise; `harmonize_tokenize_embed_pipeline` raises an error for
    `gene_perturb_df`, which it never applied (use `perturb_dataset` and
    `embed_dataset`).

-   `linear_classifier` and `linear_regressor` evaluate the weights of the
    epoch with the lowest validation loss. They kept references to the live
    parameters as the "best" state, so the last epoch was evaluated (patience
    epochs after the best one when training stopped early);
    `restore_best=False` reproduces earlier results.
-   `get_adata_cellphonedb_lr_pairs` keeps ligand-receptor pairs with
    hyphenated gene symbols (e.g. HLA genes), which were split apart and
    dropped; `legacy_split=True` restores the previous behaviour.
-   Logged training losses (per step, per epoch and in checkpoints) are
    averaged over the GPUs instead of being rank 0's own values.
-   `harmonize_tokenize_embed_pipeline` reads the gene mapping of bundles
    that ship it as `ensembl_dic.pkl` (e.g. the `v1.0` revisions on the
    Hugging Face Hub), and warns when a bundle has no gene mapping or no
    `gene_count_dictionary.pkl` (previously it silently looked the genes up
    in an Ensembl release or skipped the occurrence filter).
-   Tokenizing a directory gives the same row order in every run: files are
    tokenized in sorted path order, and in parallel mode their results are
    assembled in that order instead of the order in which they finish.
    Precomputed splits and `n_nonzero_tokens` lists refer to row numbers.
    Existing tokenized datasets are not affected.
-   Training raises instead of silently training on the wrong cells when:
    -   a precomputed split (or per-epoch split) selects cells of
        `data.test_batch_ids` / `data.val_batch_ids`. Remove the batches from
        these lists if training is meant to include them (e.g. TERRA-112M),
        or set `data.allow_heldout_in_split: true`;
    -   a precomputed `n_nonzero_tokens` list does not have one entry per
        cell of the training split;
    -   `mask.pad_masks_to_batch_max` is used without metatokens in every
        context (no special tokens, or `mask.special_token_pad_ratio` > 0)
        and without `meta.zero_fully_masked_attention_rows`: a cell with one
        or two gene tokens would have no context, and its NaN would stop
        training.
-   The launcher raises when `state.load_checkpoint` is set without
    `state.folder_path` (every launch otherwise writes to a new timestamped
    folder that holds no checkpoint), and logs the absolute run folder. A
    `read_checkpoint` without `folder_path` is read from the run folder
    (previously a `TypeError`).
-   With `keep_special_tokens=True`, inference reads and checks only the
    model's own special tokens: a model with an assay token keeps it on a
    batch it was not trained on, and TERRA-112M needs only the dataset and
    batch entries of `adata.uns`. Embeddings are unchanged.

### Fixed

-   Mixed wildcard and specific-cell requests in `perturb_dataset` now apply
    all eligible rows in their original order, including neighborhood edits;
    wildcard rows were silently skipped while flags/filtering still included
    their targets. Mixed requests support final per-segment position packing,
    and neighbor IDs are required only for specific-neighbor edits. Existing
    standalone-scope behavior and pre-edit eligibility masks are preserved.
-   A checkpoint that cannot be loaded now raises an error instead of
    silently leaving the model randomly initialized (training, finetuning and
    inference). For training, `state.allow_checkpoint_load_failure: true`
    restores the previous behaviour.
-   Resuming multi-GPU training from the latest checkpoint
    (`read_checkpoint: null`) now resumes every rank, not only rank 0. The
    latest checkpoint is written atomically.
-   Pretraining stops with a clear error, on all GPUs at the same step, at
    the first non-finite loss or gradient norm, before the optimizer step
    (the loss was checked only after the step, on each GPU separately), and
    never saves a non-finite model or optimizer state over a checkpoint.
-   Pretraining checkpoints hold the random number generator states of every
    GPU, and a resumed run restores them, so it continues the random stream
    (masks, segment counts, DataLoader worker seeds) of the interrupted run
    instead of repeating that of epoch 0; resuming from the end of an epoch
    gives the same training as an uninterrupted run, up to GPU
    nondeterminism. Checkpoints without these states are still accepted,
    with a warning. All GPUs must resume from the same epoch. A run that
    starts afresh because its checkpoint could not be loaded
    (`allow_checkpoint_load_failure`) keeps the random stream of a fresh run.
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
-   The RDA depth term is exactly zero at initialization, as documented. RoPE
    draws one rotation per forward pass, its target encoder runs in eval
    mode, and a warning flags frequency scales that alias on micrometre
    coordinates.
-   Tokenization and `harmonize_tokenize_embed_pipeline` no longer modify the
    input AnnData: the tokenizers no longer add the neighbor graph and
    normalized layers to it or log-transform a dense float32 `adata.X` in
    place, and the pipeline no longer renames its cells. Tokenizing the same
    AnnData twice gives the same result, dense and sparse input give the same
    values with shifted-log rank and count normalization, and the pipeline's
    returned and saved harmonized AnnData holds the counts. Embeddings of the
    released models are unchanged.
-   A sample with too few cells for the neighbor graph raises a clear error
    that names it.
-   Cell-neighborhood tokenization with cell-area count normalization
    normalizes the neighborhood counts instead of the index cell's counts.
-   Training with batch correction and metatoken dropout: cycle consistency
    skips the second encoder pass in batches whose padded metatokens cannot
    carry the swapped labels and checks the label range of rank models, and
    the adversarial classifier and distribution alignment pool every gene
    token of such batches.
-   Perturbing the neighborhood of specific cells edits only those cells'
    tokens in their neighbors' sequences (it edited the whole neighborhood),
    and knockouts of specific cells honour `pad_gene_tokens`.
-   `get_average_gene_embed` on CPU no longer counts cells twice when the
    data span several batches (GPU results are unchanged).
-   Embeddings of models with ALiBi position encodings include the ALiBi
    bias, as in training.
-   Checkpoints of models with read-depth-aware conditioning can be loaded
    for inference and finetuning; `tokenize_adata` reads the normalization
    files shipped in the model folder; `prepare_finetuned_model` scales the
    merged LoRA adapters by `peft_alpha / peft_rank`, as PEFT does.
-   `filter_dataset_by_cell_ids` gives every call a cache of its own, so
    repeated calls in one directory no longer return an earlier subset; the
    caches are removed when the Python session ends.
-   With `sample_key`, the pipeline returns the genes in the input order
    instead of a different order in every process.
-   The training launcher supports plain `srun` launches, and `meta.add_cls`
    is no longer silently ignored for cell-neighborhood models; the embedding
    and finetuning entry points reject models trained with it with a clear
    error instead of a weight-size mismatch.
-   `harmonize_tokenize_embed_pipeline` accepts a `pathlib.Path` save path and
    no longer derives the batch label from it; `get_gene_embed` accepts
    repeated genes.
-   `linear_regressor` reports its `JSD` metric correctly as the
    Jensen-Shannon distance (the square root of the divergence, natural
    logarithm), stores it also as `JS_distance` next to `JS_divergence`, and
    leaves cells with an all-zero composition out of the averages instead of
    returning NaN.
-   `terra.evaluation` imports without the `eval` extra; the ligand-receptor
    utilities import CellPhoneDB, OmniPath and IPython when they are called
    (importing OmniPath also sent network requests).
-   `push_model_to_hub` refuses a tag that already exists instead of silently
    leaving it on the previous revision, tags the uploaded revision exactly,
    and no longer lets a `README.md` in the bundle folder overwrite the model
    card.
-   The attention backend presets are documented and logged as PyTorch
    applies them (in its priority order, not the listed order), and TERRA
    imports on PyTorch < 2.3, where the backend cannot be selected.
-   Finetuning validation runs without autograd.

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
