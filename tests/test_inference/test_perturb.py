"""
Tests for ``perturb_dataset``.

* A ``"neighborhood"`` target on a specific cell edited the whole
  neighborhood of every cell that has it as a neighbor (all neighbor
  segments), not the cell's own tokens there. It now edits only the positions
  whose neighbor ID is that cell.
* Knockouts of specific cells kept the gene tokens (value 0) even with
  ``pad_gene_tokens=True``, unlike knockouts of all cells; both now pad them.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from terra import harmonize_adata, perturb_dataset, tokenize_adata

SEQ_LEN_CELL, N_SEGMENTS = 16, 4


def _tokenized(tiny, bundle, cache):
    harmonized = harmonize_adata(
        tiny.adata(),
        gene_mapping_dict_file_path=str(bundle / 'ensembl_dictionary.pkl'),
        min_genes_per_cell=0, min_cells_per_gene=0)
    return tokenize_adata(
        harmonized, model_folder_path=str(bundle),
        cache_directory_path=str(cache), nproc=1,
        processing_mode='sequential', add_neigh_cell_ids=True)


@pytest.fixture(scope='module')
def data(tiny, tmp_path_factory):
    """Bundles with their tokenized data."""
    out = {}
    for name, config in [('96M', None)]:
        bundle = tiny.bundle(name, config)
        out[name] = (bundle, _tokenized(
            tiny, bundle, tmp_path_factory.mktemp(f'cache_{name}')))
    return out


def _columns(dataset):
    tokens = np.stack([np.asarray(x) for x in dataset['gene_tokens']])
    values = np.stack([np.asarray(x, dtype=float)
                       for x in dataset['gene_expr']])
    return tokens, values


def _perturb(dataset, bundle, **row):
    perturb_df = pd.DataFrame([{'perturbed_ensembl_id': 'all',
                                'foldchange': np.nan, **row}])
    return perturb_dataset(
        dataset, perturb_df, model_folder_path=str(bundle),
        seq_len_cell=SEQ_LEN_CELL, n_segments=N_SEGMENTS)


def _neighbor_positions(dataset, cell_id):
    """Per cell, the positions of `cell_id`'s tokens in its neighborhood."""
    ids = np.stack([np.asarray(x) for x in dataset['cell_ids']])
    positions = ids == cell_id
    positions[:, :SEQ_LEN_CELL] = False
    return positions


def _gene_id(bundle, token):
    with open(bundle / 'token_dictionary.pkl', 'rb') as f:
        token_dict = pickle.load(f)
    return {value: key for key, value in token_dict.items()}[int(token)]


def test_neighborhood_of_a_cell_edits_only_its_tokens(data):
    bundle, dataset = data['96M']
    tokens, values = _columns(dataset)
    cell_id = dataset['cell_id'][0]
    positions = _neighbor_positions(dataset, cell_id)
    assert positions.any(axis=1).sum() > 0

    # All genes of the cell
    new_tokens, new_values = _columns(_perturb(
        dataset, bundle, perturbed_cell_id=cell_id,
        perturbation_target='neighborhood', perturbation_type='knockout'))
    np.testing.assert_array_equal(new_tokens[positions], 0)
    np.testing.assert_array_equal(new_values[positions], 0)
    np.testing.assert_array_equal(new_tokens[~positions], tokens[~positions])
    np.testing.assert_array_equal(new_values[~positions], values[~positions])

    # One gene of the cell
    token = tokens[positions][tokens[positions] != 0][0]
    edited = positions & (tokens == token)
    assert (tokens[~positions] == token).any()  # elsewhere too: kept
    new_tokens, new_values = _columns(_perturb(
        dataset, bundle, perturbed_cell_id=cell_id,
        perturbed_ensembl_id=_gene_id(bundle, token),
        perturbation_target='neighborhood', perturbation_type='knockout'))
    np.testing.assert_array_equal(new_tokens[edited], 0)
    np.testing.assert_array_equal(new_tokens[~edited], tokens[~edited])
    np.testing.assert_array_equal(new_values[~edited], values[~edited])


@pytest.mark.parametrize('cells', ['specific', 'all'])
@pytest.mark.parametrize('pad_gene_tokens', [True, False])
def test_knockouts_pad_gene_tokens_for_all_and_specific_cells(
        data, cells, pad_gene_tokens):
    bundle, dataset = data['96M']
    tokens, values = _columns(dataset)
    token = tokens[0, 0]
    cell_id = dataset['cell_id'][0] if cells == 'specific' else 'all'
    perturb_df = pd.DataFrame([{
        'perturbed_cell_id': cell_id,
        'perturbed_ensembl_id': _gene_id(bundle, token),
        'perturbation_target': 'cell', 'perturbation_type': 'knockout',
        'foldchange': np.nan}])
    new_tokens, new_values = _columns(perturb_dataset(
        dataset, perturb_df, model_folder_path=str(bundle),
        seq_len_cell=SEQ_LEN_CELL, n_segments=N_SEGMENTS,
        pad_gene_tokens=pad_gene_tokens))
    edited = np.zeros_like(tokens, dtype=bool)
    edited[:, :SEQ_LEN_CELL] = tokens[:, :SEQ_LEN_CELL] == token
    if cells == 'specific':
        edited[1:] = False
    np.testing.assert_array_equal(new_values[edited], 0)
    np.testing.assert_array_equal(
        new_tokens[edited], 0 if pad_gene_tokens else token)
    np.testing.assert_array_equal(new_tokens[~edited], tokens[~edited])
    np.testing.assert_array_equal(new_values[~edited], values[~edited])
