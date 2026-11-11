"""
Tokenizer settings recorded next to a tokenized dataset.

A tokenized dataset carries no record of how it was tokenized, so a model
config could claim settings (e.g. `data.stable_rank_ties`) that its training
data were not tokenized with; inference reads the settings from the model
config and would then tokenize new data differently from the training data.
`CellTokenizer.tokenize_data` therefore writes the settings that determine the
tokens into the saved dataset folder, and `train` checks them against the
model config and records them in it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# File written into the saved `.dataset` folder
TOKENIZER_PARAMS_FILE = 'terra_tokenizer_params.json'

# Model-config 'data' keys that must equal the dataset's tokenization
NORM_METHOD_KEYS = (
    'rank_cell_norm_method',
    'rank_gene_norm_method',
    'rank_count_norm_method',
    'count_cell_norm_method',
    'count_gene_norm_method',
    'count_count_norm_method',
)
# Optional settings that inference reads from the model config when present
# (see terra.inference.tokenize); absent keys mean the tokenizer default
OPTIONAL_KEYS = (
    'stable_rank_ties',
    'rank_tie_break_norm_factor',
    'zero_expr_from_raw_counts',
    'gcrd_basis_target_read_depth',
    'gcrd_target_read_depth_per_gene',
    'seurat_loess_span',
    'eliminate_stored_zeros',
    'pad_empty_neighbors',
    'aggregate_own_neighbors',
)
TOKENIZER_TYPES = {
    'CellGraphTokenizer': 'cell_graph',
    'CellNeighborhoodTokenizer': 'cell_neigh',
}


def _sha256(file_path: Path | str | None) -> str | None:
    if file_path is None or not os.path.exists(file_path):
        return None
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def tokenizer_params(tokenizer) -> dict:
    """Settings of `tokenizer` that determine the tokens it produces."""
    norm_methods = {key: getattr(tokenizer, key) for key in NORM_METHOD_KEYS}
    uses_pf_targets = 'pflog1ppf' in (tokenizer.rank_count_norm_method,
                                      tokenizer.count_count_norm_method)
    # The norm-factor file is read for the 'mean' / 'nonzero_mean' gene
    # normalizations and for the rank tie-break column
    uses_norm_factors = (
        tokenizer.rank_gene_norm_method in ('mean', 'nonzero_mean')
        or tokenizer.count_gene_norm_method in ('mean', 'nonzero_mean')
        or tokenizer.rank_tie_break_norm_factor is not None)
    optional = {
        'stable_rank_ties': tokenizer.stable_rank_ties,
        'rank_tie_break_norm_factor': tokenizer.rank_tie_break_norm_factor,
        'zero_expr_from_raw_counts': tokenizer.zero_expr_from_raw_counts,
        'gcrd_basis_target_read_depth':
            tokenizer.gcrd_kwargs['basis_target_read_depth'],
        'gcrd_target_read_depth_per_gene':
            tokenizer.gcrd_kwargs['target_read_depth_per_gene'],
        'seurat_loess_span': tokenizer.seurat_loess_span,
        'eliminate_stored_zeros': tokenizer.eliminate_stored_zeros,
    }
    for key in ('pad_empty_neighbors', 'aggregate_own_neighbors'):
        if hasattr(tokenizer, key):
            optional[key] = getattr(tokenizer, key)
    try:
        from terra import __version__ as terra_version
    except ImportError:
        terra_version = None
    return {
        'tokenizer': type(tokenizer).__name__,
        'terra_version': terra_version,
        'model_input_size': tokenizer.model_input_size,
        'n_neighs': tokenizer.n_neighs,
        'include_zero_expr_genes': tokenizer.include_zero_expr_genes,
        'include_special_tokens': tokenizer.include_special_tokens,
        'special_tokens': tokenizer.special_tokens,
        **norm_methods,
        **optional,
        'pf_targets_file_path': (
            str(tokenizer.pf_targets_file_path) if uses_pf_targets
            and tokenizer.pf_targets_file_path is not None else None),
        'pf_targets_sha256': (
            _sha256(tokenizer.pf_targets_file_path) if uses_pf_targets
            else None),
        'uses_pf_targets': uses_pf_targets,
        'norm_factor_file_path': (
            str(tokenizer.norm_factor_file_path) if uses_norm_factors
            else None),
        'norm_factor_sha256': (
            _sha256(tokenizer.norm_factor_file_path) if uses_norm_factors
            else None),
        'uses_norm_factors': uses_norm_factors,
        'token_dictionary_sha256': _sha256(
            tokenizer.token_dictionary_file_path),
    }


def write_tokenizer_params(tokenizer, dataset_path: Path | str):
    """Write the settings of `tokenizer` into the saved dataset folder."""
    with open(Path(dataset_path) / TOKENIZER_PARAMS_FILE, 'w') as f:
        json.dump(tokenizer_params(tokenizer), f, indent=2, sort_keys=True)


def read_tokenizer_params(dataset_path: Path | str) -> dict | None:
    """The recorded tokenizer settings of a dataset, or None if absent."""
    file_path = Path(dataset_path) / TOKENIZER_PARAMS_FILE
    if not file_path.exists():
        return None
    with open(file_path) as f:
        return json.load(f)


def reconcile_tokenizer_params(data_config: dict):
    """
    Check a model config's `data` section against the tokenizer settings
    recorded with its tokenized dataset, and record them in it.

    Raises a ValueError when the config states a setting the dataset was not
    tokenized with, or when it would make inference use other normalization
    files than the tokenization did. Optional settings the config leaves out
    are added to `data_config`, so that the saved model config makes inference
    tokenize new data as the training data were tokenized. Datasets without a
    record (tokenized before it was written) are accepted with a warning.

    Parameters
    ----------
    data_config:
        The model config's `data` section (modified in place). Its
        `norm_factor_file_path` and `pf_targets_file_path` must already be
        resolved.
    """
    dataset_path = data_config.get('tokenized_data_folder_path')
    if dataset_path is None:
        # Dataset passed in directly (e.g. tests): nothing to check against
        return
    recorded = read_tokenizer_params(dataset_path)
    if recorded is None:
        logger.warning(
            f"The dataset '{dataset_path}' has no record of its tokenizer "
            f"settings ({TOKENIZER_PARAMS_FILE}), so they cannot be checked "
            "against the model config: make sure the config's data keys "
            "(normalization methods, stable_rank_ties, pad_empty_neighbors, "
            "...) state how it was tokenized, as inference tokenizes new "
            "data with them.")
        return

    mismatches = []
    tokenizer_type = TOKENIZER_TYPES.get(recorded['tokenizer'])
    if (tokenizer_type is not None
            and data_config.get('tokenizer_type') != tokenizer_type):
        mismatches.append(
            f"tokenizer_type: config {data_config.get('tokenizer_type')!r}, "
            f"dataset {tokenizer_type!r}")
    for key in NORM_METHOD_KEYS:
        if data_config.get(key) != recorded[key]:
            mismatches.append(
                f"{key}: config {data_config.get(key)!r}, "
                f"dataset {recorded[key]!r}")
    for key in OPTIONAL_KEYS:
        if key not in recorded:
            continue
        if key in data_config and data_config[key] != recorded[key]:
            mismatches.append(
                f"{key}: config {data_config[key]!r}, "
                f"dataset {recorded[key]!r}")
    for kind, used_key in (('pf_targets', 'uses_pf_targets'),
                           ('norm_factor', 'uses_norm_factors')):
        if not recorded[used_key]:
            continue
        config_path = data_config.get(f'{kind}_file_path')
        recorded_sha = recorded[f'{kind}_sha256']
        config_sha = _sha256(config_path)
        if recorded_sha is None and config_path is not None:
            mismatches.append(
                f"{kind}_file_path: the dataset was tokenized without this "
                f"file, but the config gives '{config_path}'")
        elif recorded_sha is not None and config_sha != recorded_sha:
            mismatches.append(
                f"{kind}_file_path: config '{config_path}' "
                f"({'missing' if config_sha is None else 'other contents'}), "
                f"dataset tokenized with '{recorded[f'{kind}_file_path']}'")
    if mismatches:
        raise ValueError(
            f"The model config does not match how '{dataset_path}' was "
            "tokenized (inference tokenizes new data with the config's "
            "settings):\n  " + "\n  ".join(mismatches))

    added = [key for key in OPTIONAL_KEYS
             if key in recorded and key not in data_config]
    for key in added:
        data_config[key] = recorded[key]
    if added:
        logger.info(
            "Recorded the dataset's tokenizer settings in the model config: "
            + ", ".join(f"{key}={recorded[key]!r}" for key in added) + ".")
