"""
Tests for ``perturb_dataset``.

* A ``"neighborhood"`` target on a specific cell edited the whole
  neighborhood of every cell that has it as a neighbor (all neighbor
  segments), not the cell's own tokens there. It now edits only the positions
  whose neighbor ID is that cell.
* Knockouts of specific cells kept the gene tokens (value 0) even with
  ``pad_gene_tokens=True``, unlike knockouts of all cells; both now pad them.
* ``foldchange`` multiplied the stored values, which are log1p(counts) for
  shifted-log models (e.g. TERRA-112M/96M), so a fold change of 2 turned a
  count c into (c + 1)^2 - 1 instead of 2c. Log1p-normalized values now
  change in count space; other values are multiplied as before.
"""

import copy
import pickle

import numpy as np
import pandas as pd
import pytest
import yaml

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
    """Bundles (log1p and linear count values) with their tokenized data."""
    linear = copy.deepcopy(tiny.configs['96M'])
    linear['data']['count_count_norm_method'] = None
    out = {}
    for name, config in [('96M', None), ('96M_linear', linear)]:
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


@pytest.mark.parametrize('layout, log1p_values',
                         [('96M', True), ('96M_linear', False)])
@pytest.mark.parametrize('cells', ['specific', 'all'])
def test_fold_change_multiplies_the_counts(data, layout, log1p_values,
                                           cells):
    bundle, dataset = data[layout]
    tokens, values = _columns(dataset)
    token = tokens[0, 0]
    cell_id = dataset['cell_id'][0] if cells == 'specific' else 'all'
    new_tokens, new_values = _columns(_perturb(
        dataset, bundle, perturbed_cell_id=cell_id,
        perturbed_ensembl_id=_gene_id(bundle, token),
        perturbation_target='cell', perturbation_type='foldchange',
        foldchange=2.0))
    edited = np.zeros_like(tokens, dtype=bool)
    edited[:, :SEQ_LEN_CELL] = tokens[:, :SEQ_LEN_CELL] == token
    if cells == 'specific':
        edited[1:] = False
    if log1p_values:
        # The counts double
        np.testing.assert_allclose(np.expm1(new_values[edited]),
                                   2 * np.expm1(values[edited]), rtol=1e-5)
    else:
        np.testing.assert_allclose(new_values[edited], 2 * values[edited],
                                   rtol=1e-6)
    np.testing.assert_array_equal(new_values[~edited], values[~edited])
    np.testing.assert_array_equal(new_tokens, tokens)


def _model_folder(path, tokenizer_type):
    """Model folder with a token dictionary and a minimal model config."""
    with open(path / 'token_dictionary.pkl', 'wb') as f:
        pickle.dump({'<pad>': 0, 'ENSG1': 5, 'ENSG2': 6}, f)
    data_config = {'count_count_norm_method': 'shifted_log'}
    if tokenizer_type is not None:
        data_config['tokenizer_type'] = tokenizer_type
    with open(path / 'model_config.yaml', 'w') as f:
        yaml.safe_dump({'data': data_config}, f)
    return str(path)


@pytest.mark.parametrize('tokenizer_type', ['cell_neigh', None])
def test_cell_neighborhood_rows_with_the_equal_split(tmp_path,
                                                     tokenizer_type):
    # Cell-neighborhood rows with the equal split are two halves of
    # seq_len_cell, which n_segments does not describe: they are perturbed
    # with any n_segments (here the default, 11), as before the layout check
    from datasets import Dataset

    folder = _model_folder(tmp_path, tokenizer_type)
    tokens = np.zeros(8, np.int64)
    tokens[:2] = [5, 6]
    tokens[4:6] = [6, 5]
    values = np.zeros(8)
    values[:2] = 1.0
    values[4:6] = 2.0
    dataset = Dataset.from_dict({
        'cell_id': ['a', 'b'], 'gene_tokens': [tokens, tokens],
        'gene_expr': [values, values], 'n_nonzero_tokens': [4, 4]})
    perturb_df = pd.DataFrame([{
        'perturbed_cell_id': 'all', 'perturbed_ensembl_id': 'ENSG1',
        'perturbation_target': 'cell', 'perturbation_type': 'knockout',
        'foldchange': np.nan}])
    new_tokens, new_values = _columns(perturb_dataset(
        dataset, perturb_df, model_folder_path=folder, seq_len_cell=4))
    expected_tokens, expected_values = tokens.copy(), values.copy()
    expected_tokens[0], expected_values[0] = 0, 0.0  # index cell only
    for row in range(2):
        np.testing.assert_array_equal(new_tokens[row], expected_tokens)
        np.testing.assert_array_equal(new_values[row], expected_values)

    # Rows of a cell-graph model must be n_segments segments of seq_len_cell
    folder = _model_folder(tmp_path, 'cell_graph')
    with pytest.raises(ValueError, match='n_segments'):
        perturb_dataset(dataset, perturb_df, model_folder_path=folder,
                        seq_len_cell=4)
    assert len(perturb_dataset(dataset, perturb_df, model_folder_path=folder,
                               seq_len_cell=4, n_segments=2)) == 2
