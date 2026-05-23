import copy
import logging
import os
import pickle
import sys
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from datasets import concatenate_datasets, Dataset
from functools import partial
from tqdm import tqdm
from pyensembl import EnsemblRelease
from scipy.sparse import issparse

from app.utils import init_model, load_checkpoint, parse_protein_init_kwargs
from nichejepa.datasets.cell_datasets import CellBaseDataset, init_cell_dataset
from nichejepa.datasets.dataloaders import init_dataloader_and_sampler
from nichejepa.masks.block_masking  import BlockMaskCollator
from nichejepa.masks.cell_masking import CellMaskCollator
from nichejepa.tokenizers import cell_tokenizers
from nichejepa.utils.embedding import (create_binary_selection_mask,
                                       compute_mean_unmasked_emb,
                                       compute_unmasked_rank_based_weights,
                                       collect_adata_from_folder,
                                       retrieve_gene_emb,
                                       compute_count_mean_cosine_sim,
                                       compute_sum_and_nonzero_count,
                                       batch_rowwise_distances)
from nichejepa.utils.logging import CSVLogger
from typing import Dict, List


_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


@torch.inference_mode()
def infer(args: dict,
          dataset: CellBaseDataset,
          load_folder_path: str,
          dataset_ids: list | None = None,
          obs_cols: list | None = None,
          uns_cols: list | None = None,
          emb_layers: list | None = None,
          cell_gene_ids: list = [],
          neighborhood_gene_ids: list = [],
          agg_type: Literal['cls',
                            'avg',
                            'weighted_avg'] = 'avg',
          masked_tokens: list[int] | None = None,
          agg_excluded_genes: list[int] | None = None,
          top_k: int | None = None,
          return_gene: bool=True,
          return_cosine_sim: bool=False,
          compute_cosine_with_list:  list[str] = [],
          return_gene_per_data: bool=False,
          return_gene_marker_score: bool=False,
          return_distance: bool=False,
          include_spatial_cell_emb: bool = False,
          ignore_spc_tokens: bool = True,
          debug: bool = False,
          ) -> ad.AnnData:
    """
    Use a trained model for inference. Run forward pass on a given
    dataset andbreturn cell, neighborhood and (optionally) gene
    embeddings (cell and neighborhood gene embeddings).

    Parameters
    -----------
    args:
        Dictionary containing the hyperparameters from the config file.
    dataset:
        Cell dataset for which embeddings will be inferred.
    load_folder_path:
        Path where the checkpoint is stored.
    emb_layers:
        Layers for which to retrieve the embedding.
    cell_gene_ids:
        List with gene IDs for which cell gene embeddings will be
        retrieved.
    neighborhood_gene_ids:
        List with gene IDs for which neighborhood gene embeddings will
        be retrived.
    agg_type:
        Specifies how (aggregated) cell and neighborhood embeddings are
        computed from individual gene embeddings.
    masked_tokens:
        List of tokens to be masked by the attention mask during
        inference.
    agg_excluded_genes:
        List of tokens to be excluded from the aggregation.
    top_k:
        Include only top_k genes in aggregation.
    return_gene: 
        If 'True' will return gene_embedding.
    return_cosine_sim: 
        If 'True' will compute and return cosine_sim matrix.
    compute_cosine_with_list:
       A list that defines the items with which we want to compute cosine similarity.
       it could have value of 'cell' or/and 'neighborhood'.
    return_gene_per_data:
        If 'True' will return gene_embedding for each gene per dataset.
    return_gene_marker_score:
        If 'True' will compute and return gene marker scores.
    return_distance:
        If 'True' will compute and return distance between cosine sim of cell_neb 
        and cell_cell matrix.
    include_spatial_cell_emb:
        If 'True' also return spatial cell embedding.

    Returns
    -----------
    adata:
        An AnnData object with the stored embeddings and labels.
    """
    # Set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # Load params from config file
    add_cls = args['meta']['add_cls']
    gt_type = args['meta']['gt_type']
    if 'count_encoding' in args['meta'].keys():
        count_encoding = args['meta']['count_encoding']
    else:
        count_encoding = 'value_bins'
    if 'n_value_bins' in args['meta'].keys():
        n_value_bins = args['meta']['n_value_bins']
    else:
        n_value_bins = 100
    if 'cell_pos_enc' in args['meta'].keys():
        cell_pos_enc = args['meta']['cell_pos_enc']
    else:
        cell_pos_enc = 'segment'
    enc_depth = args['meta']['enc_depth']
    enc_emb_dim = args['meta']['enc_emb_dim']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if 'num_heads' in args['meta'].keys():
        num_heads = args['meta']['num_heads']
    else:
        num_heads = 8
    if 'mlp_ratio' in args['meta'].keys():
        mlp_ratio = args['meta']['mlp_ratio']
    else:
        mlp_ratio = 4.0
    if 'predict_gene' in args['meta'].keys():
        predict_gene = args['meta']['predict_gene']
    else:
        predict_gene = True
    if 'pos_learnable' in args['meta'].keys():
        pos_learnable = args['meta']['pos_learnable']
    else:
        pos_learnable = False
    special_tokens = args['meta']['special_tokens']
    use_bfloat16 = args['meta']['use_bfloat16']
    use_flash_attention = args['meta']['use_flash_attention']
    use_layer_norm = args['meta']['use_layer_norm']
    
    if 'api_version' in args['meta'].keys():
        api_version = args['meta']['api_version']
    else:
        api_version = 'v3'
    if 'mlp_bias' in args['meta'].keys():
        mlp_bias = args['meta']['mlp_bias']
    else:
        mlp_bias = True

    dataset_name = args['data']['dataset_name']
    token_dict_folder_path = args['data']['token_dict_folder_path']
    raw_data_folder_path = args['data']['raw_data_folder_path']
    batch_size = args['data']['batch_size']
    pin_memory = args['data']['pin_memory']
    num_workers = args['data']['num_workers']
    tokenizer_type = args['data']['tokenizer_type']
    seq_len_cell = args['data']['seq_len_cell']
    seq_len_neighborhood = args['data']['seq_len_neighborhood']
    n_segments = args['data']['n_segments']
    MAX_OCC = args['data']['n_segments'] -1 

    if 'sep_gene_tokens_neb' in args['data'].keys():
        sep_gene_tokens_neb = args['data']['sep_gene_tokens_neb']
    else:
        sep_gene_tokens_neb = False

    if 'nz_spc' in args['data'].keys():
        nz_spc = args['data']['nz_spc']
    else:
        nz_spc = False

    if 'mega_batch_mult_max' in args['data'].keys():
        mega_batch_mult_max = args['data']['mega_batch_mult_max']
    else:
        mega_batch_mult_max = 1000

    n_contexts = args['mask']['n_contexts']
    n_targets = args['mask']['n_targets']
    block_masking = args['mask']['block_masking']
    if 'cell_masking' in args['mask'].keys():
        cell_masking = args['mask']['cell_masking']
    else:
        cell_masking = False
    context_mask_size = args['mask']['context_mask_size']
    target_mask_size = args['mask']['target_mask_size']
    per_block_mask_ratio = args['mask']['per_block_mask_ratio']
    if 'restrict_special_attention' in args['meta'].keys():
        restrict_special_attention = args['meta']['restrict_special_attention']
    else:
        restrict_special_attention = False
    if 'targets_list' in args['mask'].keys():
        targets_list = args['mask']['targets_list']
    else:
        targets_list = []

    r_file = args['state']['read_checkpoint']
    tag = args['state']['write_tag']
    
    # Load token dict and get token dict-specfic params
    with open(token_dict_folder_path, 'rb') as file:
        token_dict = pickle.load(file)
    vocab_size = len(token_dict)
    if args['data'].get('n_special_values'):
        n_special_values = args['data']['n_special_values']
    else:
        n_special_values = sum(
            1 for key in token_dict if "spv" in key) # this only works now because of the dummy special values
    max_special_tokens = sum(1 for key in token_dict if "cls" in key) + sum(
        1 for key in token_dict if "spt" in key)

    if agg_excluded_genes:
        agg_excluded_tokens = [
            token_dict[gene] for gene in agg_excluded_genes]
    else:
        agg_excluded_tokens = None
    print(agg_excluded_tokens)

    # Define tokenizer-specific params
    if tokenizer_type == 'cell_neighborhood':
        if add_cls:
            special_tokens = ['cls_0', 'cls_1'] + special_tokens  
    elif tokenizer_type == 'cell_graph':
        if add_cls:
            special_tokens = [
                f'cls_{i}' for i in range(n_segments)] + special_tokens

    # Get token sequence length and number of special tokens
    n_special_tokens = len(special_tokens)
    seq_len = seq_len_cell + seq_len_neighborhood + n_special_tokens

    # Specify last emb layer if not defined
    if emb_layers is None:
        emb_layers = [enc_depth]

    # Set the folder for saving extracted features
    save_folder = f"{load_folder_path}/extracted_features"
    feature_path = f"{save_folder}/"

    os.makedirs(save_folder, exist_ok=True)
    dump = os.path.join(save_folder, f'params.yaml')
    #with open(dump, 'w') as f:
    #    yaml.dump(args, f)

    # Define checkpointing path
    latest_path = os.path.join(load_folder_path, f'{tag}-latest.pth.tar')
    load_path = (os.path.join(load_folder_path, r_file) if r_file is not None 
        else latest_path)

    # Initialize encoder, predictor, and target encoder
    target_encoder, _ = init_model(
        gt_type=gt_type,
        count_encoding=count_encoding,
        n_value_bins=n_value_bins,
        cell_pos_enc=cell_pos_enc,
        device=device,
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_special_tokens=n_special_tokens,
        n_segments=n_segments,
        n_special_values=n_special_values,
        enc_emb_dim=enc_emb_dim,
        enc_depth=enc_depth,
        pred_emb_dim=pred_emb_dim,
        pred_depth=pred_depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        use_flash_attention=use_flash_attention,
        use_layer_norm=use_layer_norm,
        api_version=api_version,
        sep_gene_tokens_neb=sep_gene_tokens_neb,
        predict_gene=predict_gene,
        pos_learnable=pos_learnable,
        nz_spc=nz_spc,
        mlp_bias=mlp_bias,
        protein_init_kwargs=parse_protein_init_kwargs(args, token_dict))

    if api_version != 'v3':
        return_layer_emb_fn = target_encoder.return_layer_emb
    else:
        return_layer_emb_fn = target_encoder.backbone.return_layer_emb

    # Initialize mask collator
    if block_masking:
       mask_collator = BlockMaskCollator(
            n_targets=n_targets,
            n_contexts=n_contexts,
            n_segments=n_segments,
            seq_len_cell=seq_len_cell,
            seq_len_neighborhood=seq_len_neighborhood,
            n_special_tokens=n_special_tokens,
            per_block_mask_ratio=per_block_mask_ratio,
            sample_segments=False,
            sample_gene_masks=False,
            restrict_special_attention=restrict_special_attention,
            special_token_pad_ratio=1.0)
    elif cell_masking:
       mask_collator = CellMaskCollator(
            n_targets=n_targets,
            n_contexts=n_contexts,
            n_segments=n_segments,
            seq_len_cell=seq_len_cell,
            seq_len_neighborhood=seq_len_neighborhood,
            n_special_tokens=n_special_tokens,
            per_block_mask_ratio=per_block_mask_ratio,
            targets_list=targets_list)

    # Initialize train and test datasets, dataloaders and samplers
    cell_dataset = init_cell_dataset(
        dataset=dataset,
        vocab_size=vocab_size,
        seq_len_cell=seq_len_cell,
        seq_len_neighborhood=seq_len_neighborhood,
        tokenizer_type=tokenizer_type,
        gt_type=gt_type,
        cell_pos_enc=cell_pos_enc,
        special_tokens=special_tokens,
        sampling_strategy=None,
        n_nonzero_tokens_list=[],
        include_cell_id=True,
        sep_gene_tokens_neb=sep_gene_tokens_neb,
        nz_spc=nz_spc)

    loader, _ = init_dataloader_and_sampler(
        cell_dataset=cell_dataset,
        batch_size=batch_size,
        distributed=False,
        world_size=1,
        rank=0,
        collate_fn=mask_collator,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=False,
        mega_batch_mult_max=mega_batch_mult_max)
    
    _, _, target_encoder, _, _, start_epoch, _ = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=None,
            predictor=None,
            target_encoder=target_encoder,
            opt=None,
            scaler=None,
            is_training=False)
    target_encoder.eval()

    # Retrieve embeddings
    all_cell_ids = []
    all_cell_emb_list = []
    if include_spatial_cell_emb:
        all_spatial_cell_emb_list = []
    all_neighborhood_emb_list = []
    all_cell_gene_emb_dict = {}
    all_neighborhood_gene_emb_dict = {}
    all_cell_gene_emb_per_data_dict = {}
    all_neighborhood_gene_emb_per_data_dict = {}
    all_cell_gene_marker_stats = {'score': [], 'pair_count': [], 'cell_count': []}
    all_neb_gene_marker_stats = {'score': [], 'pair_count': [], 'cell_count': []}
    cos_sim_dict = {}
    emd_list = []


    for itr, (udata, _, _, masks_attention, pad_special_tokens) in tqdm(enumerate(loader)):
        for key in udata.keys():
            if key != 'cell_id':
                udata[key] = udata[key].to(device, non_blocking=True)
        masks_attention = masks_attention.to(device, non_blocking=True)

        # Collect cell IDs to join metadata
        all_cell_ids.extend(udata['cell_id'])

        # Aggregate gene embeddings into cell and neighborhood
        # embeddings
        ns_tokens = udata['tokens'][:, n_special_tokens:]

        # Exclude masked tokens from aggregation
        if masked_tokens is not None:
            mask_indices = torch.isin(
                udata['tokens'],
                torch.tensor(masked_tokens, device=udata['tokens'].device)
                ).unsqueeze(1).unsqueeze(1).expand(
                    -1,-1, udata['tokens'].shape[-1], -1)
            masks_attention = masks_attention.expand(
                masks_attention.shape[0],
                masks_attention.shape[1],
                masks_attention.shape[3],
                masks_attention.shape[3])
            masks_attention[mask_indices] = 0

        # Retrieve gene embeddings from different layers
        with torch.cuda.amp.autocast(dtype=torch.bfloat16,
                                     enabled=args['meta']['use_bfloat16']):

            full_ctx, cell_only_ctx = return_layer_emb_fn(
                layers=emb_layers,
                batch=udata,
                masks_attention=masks_attention,
                need_cell_only_context=True,
                ignore_spc_tokens=ignore_spc_tokens,
            )

            cell_emb_list = []
            neighborhood_emb_list = []

            for l in emb_layers:
                cell_emb_list.append(cell_only_ctx[l].cpu())
                neighborhood_emb_list.append(full_ctx[l].cpu())

        # Aggregate gene embeddings into cell and neighborhood embeddings
        cell_mask = create_binary_selection_mask(
            ns_tokens,
            selection_type="agg_cell",
            excluded_tokens=agg_excluded_tokens,
            seq_len_cell=seq_len_cell,
            top_k=top_k).cpu()
        if tokenizer_type == 'cell_neighborhood':
            neighborhood_mask = create_binary_selection_mask(
                ns_tokens,
                selection_type="agg_neighborhood",
                excluded_tokens=agg_excluded_tokens,
                seq_len_cell=seq_len_cell,
                top_k=top_k).cpu()
        elif tokenizer_type == 'cell_graph':
            neighborhood_mask = create_binary_selection_mask(
                ns_tokens,
                selection_type="agg_graph",
                excluded_tokens=agg_excluded_tokens,
                seq_len_cell=seq_len_cell,
                top_k=top_k,
                n_segments=n_segments).cpu()

        for i, (c_emb, n_emb) in enumerate(zip(cell_emb_list, neighborhood_emb_list)):
            if (i + 1) == len(cell_emb_list) and debug:
                x = n_emb
                first_k=256
                if x.dim() != 3:
                    raise ValueError(f"Expected x to be (B,S,D), got {x.shape}")

                # Non-zero rows PER OBSERVATION: row is valid if any dim != 0
                mask_all = x.ne(0).any(dim=-1)              # (B, S)
                x_first = x[:, :first_k, :]                 # (B, K, D)
                mask_first = mask_all[:, :first_k]          # (B, K)

                # Mean over first_k valid rows
                mask_first_f = mask_first.to(x.dtype)
                sum_first = (x_first * mask_first_f.unsqueeze(-1)).sum(dim=1)        # (B, D)
                cnt_first = mask_first_f.sum(dim=1).clamp_min(1.0).unsqueeze(-1)     # (B, 1)
                mean_first = sum_first / cnt_first                                   # (B, D)

                # Mean over all valid rows
                mask_all_f = mask_all.to(x.dtype)
                sum_all = (x * mask_all_f.unsqueeze(-1)).sum(dim=1)                  # (B, D)
                cnt_all = mask_all_f.sum(dim=1).clamp_min(1.0).unsqueeze(-1)         # (B, 1)
                mean_all = sum_all / cnt_all                                         # (B, D)

                # Cosine similarity between means
                cos = F.cosine_similarity(mean_first, mean_all, dim=-1)              # (B,)

                # If either had zero valid rows originally, set cos=0
                valid = (mask_first.sum(dim=1) > 0) & (mask_all.sum(dim=1) > 0)
                cos = torch.where(valid, cos, torch.zeros_like(cos))


                print(cos)


            # Average gene embeddings into cell and neighborhood embedding 
            if agg_type == 'avg':
                cell_emb = compute_mean_unmasked_emb(c_emb, cell_mask)
                if include_spatial_cell_emb:
                    spatial_cell_emb = compute_mean_unmasked_emb(n_emb, cell_mask)
                neighborhood_emb = compute_mean_unmasked_emb(n_emb, neighborhood_mask)
                if (i + 1) == len(cell_emb_list) and debug:
                    print(neighborhood_emb.shape)
                    print(neighborhood_mask.shape)
                    print(cell_mask.sum(dim=1))
                    print(neighborhood_mask.sum(dim=1))
                    cos2 = F.cosine_similarity(spatial_cell_emb, neighborhood_emb, dim=1)
                    print(cos2)

                    nonzero_first = (n_emb[:, :256, :].abs().sum(-1) != 0)
                    nonzero_all   = (n_emb.abs().sum(-1) != 0)

                    # For cell_mask
                    cell_true_but_zero = (cell_mask[:, :256] & ~nonzero_first).float().mean()
                    cell_false_but_nonzero = ((~cell_mask[:, :256]) & nonzero_first).float().mean()

                    # For neighborhood_mask
                    neigh_true_but_zero = (neighborhood_mask & ~nonzero_all).float().mean()
                    neigh_false_but_nonzero = ((~neighborhood_mask) & nonzero_all).float().mean()

                    print("cell:   True-but-zero =", cell_true_but_zero.item(),
                        " False-but-nonzero =", cell_false_but_nonzero.item())
                    print("neigh:  True-but-zero =", neigh_true_but_zero.item(),
                        " False-but-nonzero =", neigh_false_but_nonzero.item())

                    # Token-based zero mask
                    token_zero = (ns_tokens == 0).cpu()                  # (B, S)

                    # Embedding-based zero-row mask
                    emb_zero = (n_emb.abs().sum(dim=-1) == 0).cpu()      # (B, S)
                    # or: n_emb.eq(0).all(dim=-1)

                    token_zero_but_emb_nonzero = token_zero & (~emb_zero)
                    print("token==0 but emb nonzero:",
                        token_zero_but_emb_nonzero.float().mean())

                    emb_zero_but_token_nonzero = emb_zero & (~token_zero)
                    print("emb zero but token!=0:",
                        emb_zero_but_token_nonzero.float().mean())

                    both_zero = token_zero & emb_zero
                    percentage_both_zero = both_zero.float().mean()

                    print("token==0 AND emb row == 0:", percentage_both_zero)

                    percentage_zero = (ns_tokens == 0).float().mean()
                    print("Fraction of zeros:", percentage_zero)

                    emb_zero = (n_emb.abs().sum(dim=-1) == 0)  # (B, S)

                    percentage_zero_emb_rows = emb_zero.float().mean()

                    print("Fraction of zero embedding rows:", percentage_zero_emb_rows)

                    #raise ValueError
            elif agg_type == "weighted_avg":
                cell_weights = compute_unmasked_rank_based_weights(
                    tokens, cell_mask)
                cell_emb = compute_mean_unmasked_emb(
                    c_emb * cell_weights.unsqueeze(-1),
                    cell_mask)
                if include_spatial_cell_emb:
                    spatial_cell_weights = compute_unmasked_rank_based_weights(
                        tokens, cell_mask)
                    spatial_cell_emb = compute_mean_unmasked_emb(
                        n_emb * cell_weights.unsqueeze(-1),
                        cell_mask)
                neighborhood_weights = compute_unmasked_rank_based_weights(
                    tokens, neighborhood_mask)
                neighborhood_emb = compute_mean_unmasked_emb(
                    n_emb * neighborhood_weights.unsqueeze(-1),
                    neighborhood_mask)

            # Concat layer-specific embeddings across batches
            if itr == 0:
                all_cell_emb_list.append([cell_emb])
                if include_spatial_cell_emb:
                    all_spatial_cell_emb_list.append([spatial_cell_emb])
                all_neighborhood_emb_list.append([neighborhood_emb])
            else:
                all_cell_emb_list[i].append(cell_emb)
                if include_spatial_cell_emb:
                    all_spatial_cell_emb_list[i].append(spatial_cell_emb) 
                all_neighborhood_emb_list[i].append(neighborhood_emb)

            # Store cell and neighborhood gene embeddings of last layer
            if i == (len(neighborhood_emb_list) - 1):
                emb = c_emb
                if len(cell_gene_ids) != 0 or len(neighborhood_gene_ids) != 0 :
                    if itr == 0 or itr == len(loader)-1:
                        cell_embs = torch.zeros((emb.shape[0], len(cell_gene_ids), emb.shape[-1]), device=emb.device)
                        cell_presence = torch.zeros((emb.shape[0], len(cell_gene_ids)), device=emb.device)
                    else:
                        cell_embs.zero_()
                        cell_presence.zero_()
                rows = torch.arange(emb.shape[0], device=emb.device)
                for j, gene_id in enumerate(cell_gene_ids):
                    gene_presence_local, gene_indices = retrieve_gene_emb(
                        ns_tokens=ns_tokens,
                        seq_len_cell=seq_len_cell,
                        gene_type="cell",
                        gene_id=gene_id,
                        aggregate_multiple=False
                    )
                    cell_embs[rows[gene_presence_local], j, :] = emb[rows[gene_presence_local], gene_indices[gene_presence_local], :]
                    cell_presence[:, j] = gene_presence_local.float()
                    if return_gene:
                        if itr == 0:
                            all_cell_gene_emb_dict[gene_id] = [cell_embs[:, j, :].clone()]
                        else:
                            all_cell_gene_emb_dict[gene_id].append(cell_embs[:, j, :].clone())
                    if return_gene_per_data:
                        gene_sum, gene_count = compute_sum_and_nonzero_count(cell_embs[:, j, :])
                        if itr == 0:
                            all_cell_gene_emb_per_data_dict[gene_id] = (gene_sum, gene_count)
                        else:
                            all_cell_gene_emb_per_data_dict[gene_id][0].add_(gene_sum)
                            all_cell_gene_emb_per_data_dict[gene_id][1].add_(gene_count)
                # Process neighborhood genes (multiple occurrences: compute cosine per occurrence)
                neb_occ_dict = {}
                for compute_cosine_with in compute_cosine_with_list:
                    if compute_cosine_with=='neighborhood':
                        emb = n_emb
                    neb_occ_list = []
                    neb_occ_mask_list = []
                    for j, gene_id in enumerate(neighborhood_gene_ids):
                        gene_occ, occ_mask, gene_presence_local = retrieve_gene_emb(
                            ns_tokens=ns_tokens,
                            seq_len_cell=seq_len_cell,
                            gene_type=compute_cosine_with,
                            gene_id=gene_id,
                            emb=emb,
                            aggregate_multiple=True,
                            max_occ=MAX_OCC
                        )
                        neb_occ_list.append(gene_occ)       # gene_occ: (N, max_occ, D)
                        neb_occ_mask_list.append(occ_mask)    # occ_mask: (N, max_occ)
                        if return_gene and compute_cosine_with=='neighborhood':
                            if itr == 0:
                                #all_neighborhood_gene_emb_dict[gene_id] = [gene_occ * occ_mask.unsqueeze(-1)]
                                all_neighborhood_gene_emb_dict[gene_id] = [compute_mean_unmasked_emb(gene_occ,occ_mask)]

                            else:
                                #all_neighborhood_gene_emb_dict[gene_id].append(gene_occ * occ_mask.unsqueeze(-1))
                                all_neighborhood_gene_emb_dict[gene_id].append(compute_mean_unmasked_emb(gene_occ,occ_mask))
                        if return_gene_per_data and compute_cosine_with=='neighborhood':
                            gene_sum, gene_count = compute_sum_and_nonzero_count(compute_mean_unmasked_emb(gene_occ,occ_mask))
                            if itr == 0:
                                all_neighborhood_gene_emb_per_data_dict[gene_id] = (gene_sum, gene_count)
                            else:
                                all_neighborhood_gene_emb_per_data_dict[gene_id][0].add_(gene_sum)
                                all_neighborhood_gene_emb_per_data_dict[gene_id][1].add_(gene_count)
                    # Stack neighborhood gene occurrence tensors along gene dimension:
                    # Resulting shape: (N, num_neb_genes, max_occ, D) and mask: (N, num_neb_genes, max_occ)
                    if len(neighborhood_gene_ids) != 0 and (return_cosine_sim or return_gene_marker_score or return_distance):
                        neb_occ_dict[compute_cosine_with] = (torch.stack(neb_occ_list, dim=1), torch.stack(neb_occ_mask_list, dim=1))

                # Compute cosine similarity components using our function for multiple occurrences.
                if return_cosine_sim:
                    for compute_cosine_with in compute_cosine_with_list:
                        
                        if itr == 0:
                            cos_sim_dict[compute_cosine_with] = compute_count_mean_cosine_sim(cell_embs,
                                                                                               cell_presence, 
                                                                                               neb_occ_dict[compute_cosine_with][0], 
                                                                                               neb_occ_dict[compute_cosine_with][1])
                        else:
                            sum_cos_sim_temp, pair_count_temp, cell_count_temp = compute_count_mean_cosine_sim(cell_embs,
                                                                                                                cell_presence,                                
                                                                                                                neb_occ_dict[compute_cosine_with][0],  
                                                                                                                neb_occ_dict[compute_cosine_with][1])
                            sum_cos_sim, pair_count, cell_count = cos_sim_dict[compute_cosine_with]
                            cos_sim_dict[compute_cosine_with] = (
                                sum_cos_sim + sum_cos_sim_temp,
                                pair_count + pair_count_temp,
                                cell_count + cell_count_temp
                            )
                if return_distance:
                    cos_sim_temp = []
                    for compute_cosine_with in compute_cosine_with_list:
                        sum_cos_sim, pair_count, _ = compute_count_mean_cosine_sim(
                        cell_embs, 
                        cell_presence,
                        neb_occ_dict[compute_cosine_with][0], 
                        neb_occ_dict[compute_cosine_with][1],
                        return_per_cell=True
                        )
                        cos_sim_temp.append(sum_cos_sim/pair_count)
                    _, emd_out = batch_rowwise_distances(cos_sim_temp[0], cos_sim_temp[1])
                    emd_list.append(emd_out)

                # --- Begin: gene marker score computation ---
                if return_gene_marker_score:
                    # Cell marker score
                    #sum_cos_sim, pair_count, cell_count = compute_count_mean_cosine_sim(cell_emb.unsqueeze(1), torch.ones(cell_emb.shape[0], 1), cell_embs.unsqueeze(2),cell_presence.unsqueeze(2))
                    #sum_cos_sim, pair_count, cell_count = compute_count_mean_cosine_sim(neighborhood_emb.unsqueeze(1), torch.ones(cell_emb.shape[0], 1), neb_occ_tensor, neb_occ_mask_tensor)
                    gene_score_cell, gene_score_pair_count, gene_score_cell_count = compute_count_mean_cosine_sim(
                        cell_emb.unsqueeze(1), torch.ones(cell_emb.shape[0], 1), cell_embs.unsqueeze(2), cell_presence.unsqueeze(2),
                        return_per_cell=True
                    )
                    all_cell_gene_marker_stats['score'].append(gene_score_cell.squeeze(1).cpu())
                    all_cell_gene_marker_stats['pair_count'].append(gene_score_pair_count.squeeze(1).cpu())
                    all_cell_gene_marker_stats['cell_count'].append(gene_score_cell_count.squeeze(1).cpu())
                    # Neighborhood marker score
                    gene_score_neb, gene_score_pair_count_neb, gene_score_cell_count_neb = compute_count_mean_cosine_sim(
                        neighborhood_emb.unsqueeze(1), torch.ones(neighborhood_emb.shape[0], 1),  neb_occ_dict['neighborhood'][0],
                        neb_occ_dict['neighborhood'][1], return_per_cell=True
                    )
                    all_neb_gene_marker_stats['score'].append(gene_score_neb.squeeze(1).cpu())
                    all_neb_gene_marker_stats['pair_count'].append(gene_score_pair_count_neb.squeeze(1).cpu())
                    all_neb_gene_marker_stats['cell_count'].append(gene_score_cell_count_neb.squeeze(1).cpu())
                # --- End: gene marker score computation ---
    # Add metadata
    adata = ad.AnnData(
        obs=pd.DataFrame({'cell_id': all_cell_ids},
        index=range(len(all_cell_ids))))
    print("Loading metadata AnnDatas...")
    adata_metadata = collect_adata_from_folder(
        raw_data_folder_path,
        all_cell_ids,
        dataset_ids,
        obs_cols,
        uns_cols)
    merged_obs = pd.merge(adata.obs,
                          adata_metadata.obs,
                          on='cell_id')
    adata.obs = merged_obs.set_index('cell_id')
    adata_metadata.obs = adata_metadata.obs.set_index('cell_id')
    adata_metadata = adata_metadata[adata.obs.index, :].copy()
    adata.obsm['spatial'] = adata_metadata.obsm['spatial']
   
    # Store cell and neighborhood embeddings of all observations across layers  
    for i, emb_layer in enumerate(emb_layers):

        neigh = torch.cat(all_neighborhood_emb_list[i], dim=0)        # (N, D)
        spatial = torch.cat(all_spatial_cell_emb_list[i], dim=0)      # (N, D)

        cos = F.cosine_similarity(neigh, spatial, dim=1)              # (N,)
        print(f"Layer {emb_layer} cosine:")
        print(cos)
        
        adata.obsm[f"cell_emb_layer_{emb_layer}"] = np.array(torch.cat(
            all_cell_emb_list[i],
            dim=0))
        if include_spatial_cell_emb:
            adata.obsm[f"spatial_cell_emb_layer_{emb_layer}"] = np.array(torch.cat(
                all_spatial_cell_emb_list[i],
                dim=0))            
        adata.obsm[f"neighborhood_emb_layer_{emb_layer}"] = np.array(torch.cat(
            all_neighborhood_emb_list[i],
            dim=0))

    # Store cell and neighborhood gene embeddings of all observations in the
    # last layer
    if return_gene:
        for gene_id in cell_gene_ids:
            adata.obsm[f"cell_emb_gene{gene_id}"] = np.array(torch.cat(
                all_cell_gene_emb_dict[gene_id],
                dim=0).cpu())
            del all_cell_gene_emb_dict[gene_id]

        for gene_id in neighborhood_gene_ids:
            adata.obsm[f"neighborhood_emb_gene{gene_id}"] = np.array(torch.cat(
                all_neighborhood_gene_emb_dict[gene_id],
                dim=0).cpu())
            del all_neighborhood_gene_emb_dict[gene_id]
    if return_cosine_sim:
        for compute_cosine_with in compute_cosine_with_list:
            adata.uns[f"sum_cos_sim_{compute_cosine_with}"] = cos_sim_dict[compute_cosine_with][0].numpy()
            adata.uns[f"pair_count_{compute_cosine_with}"] = cos_sim_dict[compute_cosine_with][1].numpy()
            adata.uns[f"cell_count_{compute_cosine_with}"] = cos_sim_dict[compute_cosine_with][2].numpy()
            adata.uns[f"cos_sim_{compute_cosine_with}"] = adata.uns[f"sum_cos_sim_{compute_cosine_with}"] / adata.uns[f"pair_count_{compute_cosine_with}"]
    if return_gene_per_data:
        # Concatenate features for cell and neighborhood gene embeddings
        cell_gene_emb_features = []
        cell_gene_emb_counts = []

        neighborhood_gene_emb_features = []
        neighborhood_gene_emb_counts = []

        for gene_id in cell_gene_ids:
            if gene_id in all_cell_gene_emb_per_data_dict.keys():
                sum_emb = all_cell_gene_emb_per_data_dict[gene_id][0].numpy()
                count_emb = all_cell_gene_emb_per_data_dict[gene_id][1].numpy()
                cell_gene_emb_features.append((sum_emb / count_emb).reshape(1, -1))
                cell_gene_emb_counts.append(count_emb.reshape(1, -1))

        for gene_id in neighborhood_gene_ids:
            if gene_id in all_neighborhood_gene_emb_per_data_dict.keys():
                sum_emb = all_neighborhood_gene_emb_per_data_dict[gene_id][0].numpy()
                count_emb = all_neighborhood_gene_emb_per_data_dict[gene_id][1].numpy()
                neighborhood_gene_emb_features.append((sum_emb / count_emb).reshape(1, -1))
                neighborhood_gene_emb_counts.append(count_emb.reshape(1, -1))

        # Concatenate all features, sums, and counts into single numpy arrays
        adata.uns['cell_gene_emb_average_per_data'] = np.concatenate(cell_gene_emb_features, axis=0)
        adata.uns['cell_gene_emb_counts_per_data'] = np.concatenate(cell_gene_emb_counts, axis=0)

        adata.uns['neighborhood_gene_emb_average_per_data'] = np.concatenate(neighborhood_gene_emb_features, axis=0)
        adata.uns['neighborhood_gene_emb_counts_per_data'] = np.concatenate(neighborhood_gene_emb_counts, axis=0)
    if len(cell_gene_ids) != 0:
        adata.uns['cell_gene_ids']=np.array(cell_gene_ids)
    if len(neighborhood_gene_ids) != 0:
        adata.uns['neighborhood_gene_ids']=np.array(neighborhood_gene_ids)
    # --- Begin: store gene marker score in obsm ---
    if return_gene_marker_score:
        if all_cell_gene_marker_stats['score']:
            adata.obsm['gene_marker_score_cell'] = np.array(torch.cat(all_cell_gene_marker_stats['score'], dim=0).cpu())
            adata.obsm['gene_marker_pair_count_cell'] = np.array(torch.cat(all_cell_gene_marker_stats['pair_count'], dim=0).cpu())
            adata.obsm['gene_marker_cell_count_cell'] = np.array(torch.cat(all_cell_gene_marker_stats['cell_count'], dim=0).cpu())
        if all_neb_gene_marker_stats['score']:
            adata.obsm['gene_marker_score_neb'] = np.array(torch.cat(all_neb_gene_marker_stats['score'], dim=0).cpu())
            adata.obsm['gene_marker_pair_count_neb'] = np.array(torch.cat(all_neb_gene_marker_stats['pair_count'], dim=0).cpu())
            adata.obsm['gene_marker_cell_count_neb'] = np.array(torch.cat(all_neb_gene_marker_stats['cell_count'], dim=0).cpu())
    # --- End: store gene marker score in obsm ---
    if return_distance:
       adata.obsm['emd_dist'] = np.concatenate(emd_list, axis=0)

    return adata