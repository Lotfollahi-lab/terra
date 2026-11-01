"""
Fixtures for the inference API tests: tiny model folders with the layouts of
the released models (TERRA-112M: batch metatoken, ``nz_spc``; TERRA-96M: no
metatoken, MLP count encoding) and synthetic spatial data.
"""

import copy
import pickle

import numpy as np
import pytest

N_GENES = 60

CONFIG_112M = {
    'data': {
        'dataset_name': 'toy', 'model_input_size': 64, 'n_neighs': 3,
        'rank_cell_norm_method': None, 'rank_gene_norm_method': None,
        'rank_count_norm_method': None, 'count_cell_norm_method': None,
        'count_gene_norm_method': None, 'count_count_norm_method': 'shifted_log',
        'tokenizer_type': 'cell_graph', 'seq_len_cell': 16,
        'seq_len_neighborhood': 48, 'n_segments': 4, 'sampling_strategy': None,
        'sep_gene_tokens_neb': False, 'n_special_values': 32, 'nz_spc': True},
    'meta': {
        'api_version': 'v3', 'add_cls': False, 'gt_type': 'combined',
        'count_encoding': 'value_bins', 'n_value_bins': 20,
        'cell_pos_enc': 'segment', 'enc_depth': 2, 'enc_emb_dim': 16,
        'pred_depth': 1, 'pred_emb_dim': 8, 'num_heads': 2, 'mlp_ratio': 4,
        'special_tokens': ['batch'], 'pos_learnable': False,
        'use_bfloat16': False, 'use_flash_attention': True,
        'use_layer_norm': True, 'predict_gene': True, 'new_spc': True,
        'restrict_special_attention': False},
    'mask': {'n_contexts': 1, 'n_targets': 1, 'block_masking': True,
             'cell_masking': False, 'context_mask_size': 200,
             'target_mask_size': 50, 'per_block_mask_ratio': 0.6},
}
# Like the released config: no data.nz_spc / n_special_values / meta.new_spc
CONFIG_96M = copy.deepcopy(CONFIG_112M)
CONFIG_96M['data'].pop('n_special_values')
CONFIG_96M['data'].pop('nz_spc')
CONFIG_96M['meta'].update({'special_tokens': [], 'count_encoding': 'mlp',
                           'cell_pos_enc': 'coord'})
CONFIG_96M['meta'].pop('new_spc')

# Metadata of a section whose batch the model was trained with
CORPUS_UNS = {'dataset_id': 'ds1', 'batch': 'b0', 'assay': 'xenium',
              'species': 'human', 'tissue': 'lung'}


def make_token_dict():
    """Token dictionary with special tokens at the ids the datasets expect."""
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
    for key in ['spv_ds1_b0', 'spv_ds1_b1', 'spv_xenium', 'spv_human',
                'spv_lung', f'spv_gene_panel{N_GENES}']:
        token_dict[key] = len(token_dict)
    return token_dict, genes


def write_bundle(folder, config, seed=0):
    """Write a model folder whose checkpoint has the architecture of `config`."""
    import torch
    import yaml

    from terra.utils.helper import init_model_from_config

    folder.mkdir(parents=True)
    config = copy.deepcopy(config)
    token_dict, genes = make_token_dict()
    with open(folder / 'token_dictionary.pkl', 'wb') as f:
        pickle.dump(token_dict, f)
    with open(folder / 'ensembl_dictionary.pkl', 'wb') as f:
        pickle.dump({f'GENE{i}': gene for i, gene in enumerate(genes)}, f)
    with open(folder / 'gene_count_dictionary.pkl', 'wb') as f:
        pickle.dump({gene: 1000 for gene in genes}, f)
    with open(folder / 'model_config.yaml', 'w') as f:
        yaml.safe_dump(config, f)
    torch.manual_seed(seed)
    encoder, predictor = init_model_from_config(config, token_dict, 'cpu')
    # Move every weight away from its initialization
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for param in encoder.parameters():
            param.add_(0.1 * torch.randn(param.shape, generator=generator))
    # DDP-style keys, as saved in training
    encoder_state = {f'module.{k}': v for k, v in encoder.state_dict().items()}
    predictor_state = {
        f'module.{k}': v for k, v in predictor.state_dict().items()}
    torch.save({'encoder': encoder_state, 'predictor': predictor_state,
                'target_encoder': encoder_state, 'epoch': 0},
               folder / 'model_checkpoint.pt')
    return folder


def make_adata(n_cells=40, seed=0, uns=None):
    """Synthetic spatial AnnData with raw counts and gene symbols."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    adata = ad.AnnData(
        X=rng.poisson(2.0, size=(n_cells, N_GENES)).astype(np.float32))
    adata.var_names = [f'GENE{i}' for i in range(N_GENES)]
    adata.obs_names = [f'c{i}' for i in range(n_cells)]
    adata.obs['sample'] = 'S1'
    adata.obsm['spatial'] = rng.uniform(
        0, 1000, size=(n_cells, 2)).astype(np.float32)
    if uns:
        adata.uns.update(uns)
    return adata


def tokenize(bundle, cache, adata, include_special_tokens=False):
    """Harmonize and tokenize `adata` for the model in `bundle`."""
    from terra import harmonize_adata, tokenize_adata

    harmonized = harmonize_adata(
        adata,
        gene_mapping_dict_file_path=str(bundle / 'ensembl_dictionary.pkl'),
        min_genes_per_cell=0, min_cells_per_gene=0)
    return tokenize_adata(
        harmonized, model_folder_path=str(bundle),
        cache_directory_path=str(cache), nproc=1,
        processing_mode='sequential',
        include_special_tokens=include_special_tokens)


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    """
    Factory for model folders and tokenized data, cached per test module.

    ``tiny.bundle(name, config)`` writes (once) and returns a model folder;
    ``tiny.dataset(bundle, include_special_tokens=False, seed=0)`` returns the
    tokenized synthetic data for it; ``tiny.adata(...)`` returns new
    synthetic data (see ``make_adata``).
    """
    root = tmp_path_factory.mktemp('tiny')

    class Tiny:
        configs = {'112M': CONFIG_112M, '96M': CONFIG_96M}
        uns = CORPUS_UNS

        def __init__(self):
            self._bundles = {}
            self._datasets = {}

        def bundle(self, name, config=None):
            if name not in self._bundles:
                self._bundles[name] = write_bundle(
                    root / name, config or self.configs[name])
            return self._bundles[name]

        @staticmethod
        def adata(n_cells=40, seed=0, uns=None):
            return make_adata(n_cells=n_cells, seed=seed, uns=uns)

        def dataset(self, bundle, include_special_tokens=False, seed=0,
                    n_cells=40):
            key = (str(bundle), include_special_tokens, seed, n_cells)
            if key not in self._datasets:
                cache = root / f'cache{len(self._datasets)}'
                self._datasets[key] = tokenize(
                    bundle, cache,
                    make_adata(n_cells=n_cells, seed=seed, uns=CORPUS_UNS),
                    include_special_tokens=include_special_tokens)
            return self._datasets[key]

    return Tiny()
