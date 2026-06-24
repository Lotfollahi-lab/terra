# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.1.0]

First public release of TERRA, a JEPA-based foundation model for spatial
transcriptomics.

### Added

-   Pretrained model loading and publishing via the Hugging Face Hub
    (`download_pretrained`, `terra-hub`), with self-contained model bundles that
    include the gene-reference files needed for harmonization.
-   Inference pipeline — `harmonize_tokenize_embed_pipeline` and the individual
    `harmonize_adata` / `tokenize_adata` / `embed_dataset` steps — producing
    cell- and neighbourhood-level embeddings.
-   Downstream analysis utilities: gene embeddings, spatial gene-pair scoring,
    EMD distance, and in-silico perturbation.
-   Documentation, end-to-end tutorial, and API reference.
