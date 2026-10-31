"""
Tests for ``BlockMaskCollator(pad_masks_to_batch_max=True)``.

By default the collator trims every cell's context and target index lists to
the shortest list in the batch, so the cell with the fewest gene tokens sets
the context/target size of all cells. With ``pad_masks_to_batch_max`` the
lists are padded to the longest one and flagged by validity masks, which the
training loop applies as attention key masks (``padding_key_masks``) and to
the loss. Padded entries must then not change any valid output.
"""

import pickle

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from terra.datasets.cell_datasets import init_cell_dataset
from terra.masks.block_masking import BlockMaskCollator
from terra.masks.utils import padding_key_masks
from terra.tokenizers import CellGraphTokenizer
from terra.utils.helper import init_model

N_CELLS, N_GENES, N_NEIGHS, SEQ_LEN_CELL = 12, 40, 3, 8
MODEL_INPUT_SIZE = SEQ_LEN_CELL * (N_NEIGHS + 1)
N_SPECIAL_TOKENS = 1  # batch metatoken, as in TERRA-112M


@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('padded_masks')
    gene_ids = [f'ENSG{i:011d}' for i in range(N_GENES)]
    token_dict = {'<pad>': 0, '<mask>': 1}
    token_dict.update({g: i + 2 for i, g in enumerate(gene_ids)})
    for key in ('spv_0_batch0', f'spv_gene_panel{N_GENES}', 'spv_xenium',
                'spv_homo_sapiens', 'spv_lung'):
        token_dict[key] = len(token_dict)
    token_dictionary_file_path = tmp_path / 'token_dictionary.pkl'
    with open(token_dictionary_file_path, 'wb') as f:
        pickle.dump(token_dict, f)

    # Cells with very different numbers of expressed genes
    rng = np.random.default_rng(0)
    lam = np.repeat([0.05, 0.2, 1.0], N_CELLS // 3)[:, None]
    counts = rng.poisson(lam, (N_CELLS, N_GENES)).astype(np.float32)
    counts[counts.sum(axis=1) == 0, 0] = 1
    adata = ad.AnnData(X=sp.csr_matrix(counts))
    adata.var['ensembl_id'] = gene_ids
    adata.obs['cell_id'] = [f'cell_{i}' for i in range(N_CELLS)]
    adata.obsm['spatial'] = rng.uniform(0, 100, (N_CELLS, 2))
    adata.uns.update(dataset_id=0, batch='batch0', assay='xenium',
                     species='homo_sapiens', tissue='lung')

    tk = CellGraphTokenizer(
        model_input_size=MODEL_INPUT_SIZE, n_neighs=N_NEIGHS,
        rank_cell_norm_method=None, rank_gene_norm_method=None,
        rank_count_norm_method=None, count_cell_norm_method=None,
        count_gene_norm_method=None, count_count_norm_method='shifted_log',
        token_dictionary_file_path=token_dictionary_file_path,
        include_special_tokens=True)
    hf_dataset = tk._create_dataset(
        dataset_dict=tk._tokenize_adata(adata=adata),
        cache_directory_path=str(tmp_path))
    columns = [c for c in hf_dataset.features if c != 'cell_id']
    hf_dataset.set_format(type='torch', columns=columns,
                          output_all_columns=True)
    return init_cell_dataset(
        dataset=hf_dataset, vocab_size=len(token_dict),
        seq_len_cell=SEQ_LEN_CELL,
        seq_len_neighborhood=MODEL_INPUT_SIZE - SEQ_LEN_CELL,
        tokenizer_type='cell_graph', gt_type='combined',
        cell_pos_enc='segment', special_tokens=['batch'],
        sampling_strategy=None, n_nonzero_tokens_list=[],
        include_cell_id=False, sep_gene_tokens_neb=False, nz_spc=True)


def _collate(dataset, pad):
    collator = BlockMaskCollator(
        n_targets=1, n_contexts=1, n_segments=N_NEIGHS + 1,
        seq_len_cell=SEQ_LEN_CELL,
        seq_len_neighborhood=MODEL_INPUT_SIZE - SEQ_LEN_CELL,
        n_special_tokens=N_SPECIAL_TOKENS, per_block_mask_ratio=0.6,
        sample_segments=False, sample_gene_masks=True,
        pad_masks_to_batch_max=pad)
    torch.manual_seed(0)
    return collator([dataset[i] for i in range(len(dataset))])


def test_padded_masks_hold_every_sampled_index(dataset):
    _, ctx_trim, tgt_trim, _, _ = _collate(dataset, pad=False)
    udata, ctx, tgt, _, _ = _collate(dataset, pad=True)
    ctx_valid = udata['masks_enc_valid']
    tgt_valid = udata['masks_pred_valid']
    assert ctx.shape == ctx_valid.shape and tgt.shape == tgt_valid.shape
    # Lists are padded to the longest cell, not trimmed to the shortest
    assert ctx.shape[-1] == ctx_valid.sum(-1).max() > ctx_trim.shape[-1]
    assert tgt.shape[-1] == tgt_valid.sum(-1).max() > tgt_trim.shape[-1]
    # Same RNG: the trimmed masks are prefixes of the padded masks
    torch.testing.assert_close(ctx[..., :ctx_trim.shape[-1]], ctx_trim)
    torch.testing.assert_close(tgt[..., :tgt_trim.shape[-1]], tgt_trim)
    # Valid entries: the special token plus every non-zero token, either as
    # context or as target, exactly once
    tokens = udata['tokens']
    for b in range(tokens.shape[0]):
        c = ctx[0, b][ctx_valid[0, b]]
        t = tgt[0, b][tgt_valid[0, b]]
        assert c[0] == t[0] == 0  # special token first
        genes = torch.cat([c[1:], t[1:]]).sort().values
        torch.testing.assert_close(genes, tokens[b].nonzero().squeeze(-1)[1:])


@pytest.mark.parametrize('new_spc', [True, False])
def test_padded_entries_do_not_change_valid_outputs(dataset, new_spc):
    udata, ctx, tgt, _, _ = _collate(dataset, pad=True)
    ctx_valid = udata.pop('masks_enc_valid')[0]
    tgt_valid = udata.pop('masks_pred_valid')[0]
    torch.manual_seed(0)
    encoder, predictor = init_model(
        gt_type='combined', count_encoding='value_bins', n_value_bins=20,
        cell_pos_enc='segment', device='cpu',
        vocab_size=128,  # the dataset uses token id 106 for the batch token
        seq_len=MODEL_INPUT_SIZE + N_SPECIAL_TOKENS,
        n_special_tokens=N_SPECIAL_TOKENS, n_segments=N_NEIGHS + 1,
        n_special_values=16, enc_emb_dim=32, enc_depth=2, pred_emb_dim=16,
        pred_depth=2, num_heads=2, use_flash_attention=True,
        api_version='v3', new_spc=new_spc, nz_spc=True)
    encoder.eval()
    predictor.eval()
    enc_mask, pred_mask = padding_key_masks(
        ctx_valid, tgt_valid, n_special_tokens=N_SPECIAL_TOKENS,
        new_spc=new_spc)

    with torch.no_grad():
        z, token_emb = encoder(batch=udata, masks=[ctx[0]],
                               masks_attention=enc_mask)
        pred = predictor(z=z, token_emb=token_emb, batch=udata,
                         masks_enc=[ctx[0]], masks_pred=[tgt[0]],
                         masks_attention=pred_mask)[0]

        # Each cell on its own with its unpadded masks
        for b in range(ctx.shape[1]):
            batch_b = {k: v[b:b + 1] for k, v in udata.items()}
            ctx_b = ctx[0, b:b + 1][:, ctx_valid[b]]
            tgt_b = tgt[0, b:b + 1][:, tgt_valid[b]]
            z_b, token_emb_b = encoder(batch=batch_b, masks=[ctx_b],
                                       masks_attention=None)
            pred_b = predictor(z=z_b, token_emb=token_emb_b, batch=batch_b,
                               masks_enc=[ctx_b], masks_pred=[tgt_b],
                               masks_attention=None)[0]
            torch.testing.assert_close(z[0][b, :ctx_b.shape[1]], z_b[0][0],
                                       rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(pred[b, :tgt_b.shape[1]], pred_b[0],
                                       rtol=1e-4, atol=1e-5)
