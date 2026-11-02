"""
Tests for ``harmonize_tokenize_embed_pipeline`` on input with cells without
genes, on a tiny, randomly initialized model bundle (CPU only, no downloads).

Cells without any expressed gene have no tokens of their own: their cell
embeddings are all-zero (NaN with softmax aggregation; e.g. the TERRA-112M
quickstart printed a NaN cell embedding cosine similarity), and they shift the
neighbor segments of the cells around them. The default
``min_genes_per_cell=0`` keeps them, which reproduces existing results, and
the tokenizer warns; ``min_genes_per_cell=1`` removes them. Data without such
cells is embedded identically either way.
"""

import pickle
import warnings

import anndata as ad
import numpy as np
import pytest
import torch
import yaml

from terra import harmonize_tokenize_embed_pipeline
from terra.utils.helper import init_model_from_config

N_GENES = 40
N_CELLS = 48
EMPTY_CELLS = (5, 30)  # one in each sample
EMBEDDINGS = ('cell_emb', 'spatial_cell_emb', 'neighborhood_emb')

# TERRA-112M layout (cell graph, segment encoding, batch metatoken), scaled
# down
CONFIG = {
    'data': {
        'dataset_name': 'toy', 'model_input_size': 64, 'n_neighs': 3,
        'rank_cell_norm_method': None, 'rank_gene_norm_method': None,
        'rank_count_norm_method': None, 'count_cell_norm_method': None,
        'count_gene_norm_method': None, 'count_count_norm_method': 'shifted_log',
        'tokenizer_type': 'cell_graph', 'seq_len_cell': 16,
        'seq_len_neighborhood': 48, 'n_segments': 4, 'sampling_strategy': None,
        'sep_gene_tokens_neb': False, 'n_special_values': 32, 'nz_spc': True},
    'meta': {
        'api_version': 'v3', 'gt_type': 'combined', 'count_encoding': 'value_bins',
        'n_value_bins': 20, 'cell_pos_enc': 'segment', 'enc_depth': 1,
        'enc_emb_dim': 16, 'pred_depth': 1, 'pred_emb_dim': 8, 'num_heads': 2,
        'mlp_ratio': 4, 'special_tokens': ['batch'], 'pos_learnable': False,
        'use_bfloat16': False, 'use_flash_attention': True,
        'use_layer_norm': True, 'predict_gene': True, 'new_spc': True,
        'restrict_special_attention': False},
    'mask': {'n_contexts': 1, 'n_targets': 1, 'per_block_mask_ratio': 0.6},
}


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    folder = tmp_path_factory.mktemp('bundle')
    # Special tokens at the ids the cell datasets expect, then the genes
    token_dict = {'<pad>': 0}
    for i in range(11):
        token_dict[f'cls_{i}'] = len(token_dict)
    while len(token_dict) < 103:
        token_dict[f'reserved_{len(token_dict)}'] = len(token_dict)
    for key in ['spt_tissue', 'spt_assay', 'spt_gene_panel', 'spt_batch']:
        token_dict[key] = len(token_dict)
    genes = [f'ENSG{i + 1:011d}' for i in range(N_GENES)]
    for gene in genes:
        token_dict[gene] = len(token_dict)
    token_dict['spv_toy'] = len(token_dict)
    with open(folder / 'token_dictionary.pkl', 'wb') as f:
        pickle.dump(token_dict, f)
    with open(folder / 'ensembl_dictionary.pkl', 'wb') as f:
        pickle.dump({f'GENE{i}': gene for i, gene in enumerate(genes)}, f)
    with open(folder / 'gene_count_dictionary.pkl', 'wb') as f:
        pickle.dump({gene: 1000 for gene in genes}, f)
    with open(folder / 'model_config.yaml', 'w') as f:
        yaml.safe_dump(CONFIG, f)
    torch.manual_seed(0)
    encoder, predictor = init_model_from_config(CONFIG, token_dict, 'cpu')
    # DDP-style keys, as saved in training
    encoder_state = {f'module.{k}': v for k, v in encoder.state_dict().items()}
    predictor_state = {
        f'module.{k}': v for k, v in predictor.state_dict().items()}
    torch.save({'encoder': encoder_state, 'predictor': predictor_state,
                'target_encoder': encoder_state, 'epoch': 0},
               folder / 'model_checkpoint.pt')
    return folder


def _adata(empty_cells=()):
    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    counts[list(empty_cells)] = 0
    adata = ad.AnnData(X=counts)
    adata.var_names = [f'GENE{i}' for i in range(N_GENES)]
    adata.obs_names = [f'c{i}' for i in range(N_CELLS)]
    adata.obs['sample'] = np.repeat(['S1', 'S2'], N_CELLS // 2)
    adata.obsm['spatial'] = rng.uniform(
        0, 1000, size=(N_CELLS, 2)).astype(np.float32)
    return adata


def _embed(adata, bundle, cache, **kwargs):
    """Run the pipeline; return its result and the messages of its
    UserWarnings."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        result = harmonize_tokenize_embed_pipeline(
            adata=adata, batch_key='batch', model_folder_path=str(bundle),
            cache_directory_path=str(cache), nproc=1,
            processing_mode='sequential', num_workers=0, batch_size=16,
            **kwargs)
    messages = [str(record.message) for record in records
                if record.category is UserWarning]
    return result, messages


def _empty_cell_warnings(messages):
    return [message for message in messages if 'no expressed gene' in message]


@pytest.mark.parametrize('sample_key', [None, 'sample'])
@pytest.mark.parametrize('agg_type', ['avg', 'softmax'])
def test_cells_without_genes_are_kept_by_default_with_a_warning(
        bundle, tmp_path, sample_key, agg_type):
    result, messages = _embed(_adata(EMPTY_CELLS), bundle, tmp_path,
                              sample_key=sample_key, agg_type=agg_type)
    # One warning per tokenized AnnData (here: per sample) with such cells
    warned = _empty_cell_warnings(messages)
    assert len(warned) == (2 if sample_key else 1)
    assert all('min_genes_per_cell >= 1' in message for message in warned)

    # The cells stay, with empty cell embeddings, as before
    assert list(result.obs_names) == [f'c{i}' for i in range(N_CELLS)]
    empty = result[[f'c{i}' for i in EMPTY_CELLS]]
    for key in ('cell_emb', 'spatial_cell_emb'):
        if agg_type == 'avg':
            assert np.all(empty.obsm[key] == 0), key
        else:
            assert np.all(np.isnan(empty.obsm[key])), key


@pytest.mark.parametrize('sample_key', [None, 'sample'])
@pytest.mark.parametrize('agg_type', ['avg', 'softmax'])
def test_min_genes_per_cell_1_removes_cells_without_genes(
        bundle, tmp_path, sample_key, agg_type):
    result, messages = _embed(_adata(EMPTY_CELLS), bundle, tmp_path,
                              sample_key=sample_key, agg_type=agg_type,
                              min_genes_per_cell=1)
    assert not _empty_cell_warnings(messages)
    assert result.n_obs == N_CELLS - len(EMPTY_CELLS)
    assert not set(result.obs_names) & {f'c{i}' for i in EMPTY_CELLS}
    for key in EMBEDDINGS:
        norms = np.linalg.norm(result.obsm[key], axis=1)
        assert np.all(np.isfinite(norms)) and np.all(norms > 0), key


@pytest.mark.parametrize('sample_key', [None, 'sample'])
def test_data_with_genes_in_every_cell_is_identical_either_way(
        bundle, tmp_path, sample_key):
    default, messages = _embed(_adata(), bundle, tmp_path / 'default',
                               sample_key=sample_key)
    assert not _empty_cell_warnings(messages)
    filtered, _ = _embed(_adata(), bundle, tmp_path / 'filtered',
                         sample_key=sample_key, min_genes_per_cell=1)
    assert list(default.obs_names) == list(filtered.obs_names)
    for key in EMBEDDINGS:
        np.testing.assert_array_equal(default.obsm[key], filtered.obsm[key])
