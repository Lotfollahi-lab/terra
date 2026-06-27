"""TERRA: a foundation model for spatial transcriptomics."""

import logging as _logging

# Library best practice: attach a NullHandler so importing TERRA never forces
# logging output nor emits "no handler" warnings. Applications and CLIs opt in
# to seeing logs via e.g. ``logging.basicConfig(level="INFO")``.
_logging.getLogger("terra").addHandler(_logging.NullHandler())

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("terra-st")
except _PackageNotFoundError:  # pragma: no cover - running from an uninstalled source tree
    __version__ = "0.0.0"

from . import (datasets,
               masks,
               models,
               preprocessors,
               tokenizers,
               utils,
               inference)
from .hub import download_pretrained, push_model_to_hub
from .inference import (embed_dataset,
                        gene_embed_dataset,
                        get_average_gene_embed,
                        get_emd_distance,
                        get_gene_embed,
                        get_spatial_score,
                        harmonize_adata,
                        harmonize_tokenize_embed_pipeline,
                        infer,
                        perturb_dataset,
                        tokenize_adata)

# Top-level public API. The training application layer is available as the
# `terra.training` subpackage (imported on demand, e.g. `python -m
# terra.training.main`) and is intentionally not eagerly imported here.
__all__ = [
    "harmonize_tokenize_embed_pipeline",
    "harmonize_adata",
    "tokenize_adata",
    "embed_dataset",
    "gene_embed_dataset",
    "get_gene_embed",
    "get_average_gene_embed",
    "infer",
    "perturb_dataset",
    "get_emd_distance",
    "get_spatial_score",
    "download_pretrained",
    "push_model_to_hub",
]
