"""
Inference on pre-tokenized compact datasets (``storage_format='compact'``):
``embed_dataset`` and ``infer`` must return the same embeddings, bit for bit,
as on the padded tokenization of the same data, including models that read a
smaller layout of the dataset (``truncate_neighbors`` /
``tokenized_seq_len_cell``).
"""

import copy

import numpy as np
import pytest

from terra import embed_dataset, infer
from terra.datasets.compact import is_compact
from terra.tokenizers import cell_tokenizers

KW = dict(batch_size=8, num_workers=0)


class _CompactCellGraphTokenizer(cell_tokenizers.CellGraphTokenizer):
    def __init__(self, **kwargs):
        super().__init__(storage_format='compact', **kwargs)


def _tokenized_pair(tiny, bundle, monkeypatch, include_special_tokens):
    """Padded and compact tokenizations of the same data for `bundle`."""
    from terra import tokenize_adata

    harmonized = tiny.harmonized(bundle, include_special_tokens)
    padded = tiny.dataset(bundle, include_special_tokens)
    with monkeypatch.context() as m:
        m.setattr(cell_tokenizers, 'CellGraphTokenizer',
                  _CompactCellGraphTokenizer)
        compact = tokenize_adata(
            harmonized.copy(), model_folder_path=str(bundle),
            cache_directory_path=str(bundle.parent / f'cache_c{bundle.name}'),
            nproc=1, processing_mode='sequential',
            include_special_tokens=include_special_tokens)
    assert is_compact(compact) and not is_compact(padded)
    return padded, compact


def _assert_equal_outputs(a, b):
    assert a.keys() == b.keys()
    for key in a:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        assert x.dtype == y.dtype and x.shape == y.shape, key
        assert np.array_equal(x, y), key


@pytest.mark.parametrize('layout,special', [('112M', True), ('112M', False),
                                            ('96M', False)])
def test_embed_dataset_equal(tiny, monkeypatch, layout, special):
    bundle = tiny.bundle(layout)
    padded, compact = _tokenized_pair(tiny, bundle, monkeypatch, special)
    for kwargs in (dict(), dict(return_token_embeddings=True),
                   dict(keep_special_tokens=special)):
        _assert_equal_outputs(embed_dataset(padded, str(bundle), **KW, **kwargs),
                              embed_dataset(compact, str(bundle), **KW,
                                            **kwargs))


def test_embed_dataset_equal_sliced_layout(tiny, monkeypatch):
    # A model reading 3 segments x 8 genes of data tokenized with 4 x 16
    corpus_bundle = tiny.bundle('112M')
    padded, compact = _tokenized_pair(tiny, corpus_bundle, monkeypatch, True)
    config = copy.deepcopy(tiny.configs['112M'])
    config['data'].update({
        'model_input_size': 24, 'n_neighs': 2, 'seq_len_cell': 8,
        'seq_len_neighborhood': 16, 'n_segments': 3,
        'truncate_neighbors': True, 'tokenized_seq_len_cell': 16})
    bundle = tiny.bundle('112M_sliced', config)
    for kwargs in (dict(), dict(return_token_embeddings=True)):
        _assert_equal_outputs(embed_dataset(padded, str(bundle), **KW, **kwargs),
                              embed_dataset(compact, str(bundle), **KW,
                                            **kwargs))


@pytest.mark.parametrize('layout', ['112M', '96M'])
def test_infer_equal(tiny, monkeypatch, tmp_path, layout):
    config = tiny.configs[layout]
    bundle = tiny.bundle(layout)
    padded, compact = _tokenized_pair(tiny, bundle, monkeypatch, False)
    raw = tmp_path / 'raw'
    raw.mkdir()
    tiny.harmonized(bundle).write_h5ad(raw / 'S1.h5ad')
    args = copy.deepcopy(config)
    args['data'].update({
        'token_dict_folder_path': str(bundle / 'token_dictionary.pkl'),
        'raw_data_folder_path': str(raw), 'batch_size': 8,
        'pin_memory': False, 'num_workers': 0})
    args['state'] = {'read_checkpoint': 'model_checkpoint.pt',
                     'write_tag': 'terra'}
    a = infer(args=copy.deepcopy(args), dataset=padded,
              load_folder_path=str(bundle), include_spatial_cell_emb=True)
    b = infer(args=copy.deepcopy(args), dataset=compact,
              load_folder_path=str(bundle), include_spatial_cell_emb=True)
    assert list(a.obs_names) == list(b.obs_names)
    assert a.obsm.keys() == b.obsm.keys()
    for key in a.obsm:
        assert np.array_equal(np.asarray(a.obsm[key]),
                              np.asarray(b.obsm[key])), key


def test_perturb_dataset_equal(tiny, monkeypatch, tmp_path):
    # perturb_dataset converts a compact dataset to the padded layout first
    import pandas as pd

    from terra import harmonize_adata, perturb_dataset, tokenize_adata

    bundle = tiny.bundle('96M')
    harmonized = harmonize_adata(
        tiny.adata(),
        gene_mapping_dict_file_path=str(bundle / 'ensembl_dictionary.pkl'),
        min_genes_per_cell=0, min_cells_per_gene=0)
    out = {}
    for fmt in ('padded', 'compact'):
        with monkeypatch.context() as m:
            if fmt == 'compact':
                m.setattr(cell_tokenizers, 'CellGraphTokenizer',
                          _CompactCellGraphTokenizer)
            out[fmt] = tokenize_adata(
                harmonized.copy(), model_folder_path=str(bundle),
                cache_directory_path=str(tmp_path / f'cache_{fmt}'), nproc=1,
                processing_mode='sequential', add_neigh_cell_ids=True)
    assert is_compact(out['compact'])
    padded = out['padded']
    tokens = np.stack([np.asarray(x) for x in padded['gene_tokens']])
    with open(bundle / 'token_dictionary.pkl', 'rb') as f:
        import pickle
        id_to_gene = {v: k for k, v in pickle.load(f).items()}
    gene = id_to_gene[int(tokens[0, 0])]
    perturb_df = pd.DataFrame([
        {'perturbed_ensembl_id': gene, 'perturbed_cell_id': 'all',
         'perturbation_target': 'cell', 'perturbation_type': 'foldchange',
         'foldchange': 2.0},
        {'perturbed_ensembl_id': gene,
         'perturbed_cell_id': padded['cell_id'][3],
         'perturbation_target': 'neighborhood',
         'perturbation_type': 'knockout', 'foldchange': np.nan}])
    results = [perturb_dataset(
        dataset, perturb_df, model_folder_path=str(bundle), seq_len_cell=16,
        n_segments=4, adjust_positions=True,
        return_perturbation_flags=True).with_format(None)
        for dataset in (padded, out['compact'])]
    a, b = results
    assert a.column_names == b.column_names and len(a) == len(b)
    for col in a.column_names:
        assert a[col] == b[col], col
