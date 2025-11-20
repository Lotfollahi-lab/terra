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

from app.helper import init_model, load_checkpoint
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
          agg_excluded_tokens: list[int] | None = None,
          top_k: int | None = None,
          return_gene: bool=True,
          return_cosine_sim: bool=False,
          compute_cosine_with_list:  list[str] = [],
          return_gene_per_data: bool=False,
          return_gene_marker_score: bool=False,
          return_distance: bool=False,
          include_spatial_cell_emb: bool = False,
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
    agg_excluded_tokens:
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
    
    if 'api_version' in args['meta'].keys():
        api_version = args['meta']['api_version']
    else:
        api_version = 'v3'

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
    n_special_values = sum(1 for key in token_dict if "spv" in key)
    max_special_tokens = sum(1 for key in token_dict if "cls" in key) + sum(
        1 for key in token_dict if "spt" in key)

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
        api_version=api_version,
        sep_gene_tokens_neb=sep_gene_tokens_neb,
        predict_gene=predict_gene,
        pos_learnable=pos_learnable)

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
            sample_gene_masks=False)
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
        sep_gene_tokens_neb=sep_gene_tokens_neb)

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
        persistent_workers=False)
    
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


    for itr, (udata, _, _, masks_attention) in tqdm(enumerate(loader)):
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
            # Average gene embeddings into cell and neighborhood embedding 
            if agg_type == 'avg':
                cell_emb = compute_mean_unmasked_emb(c_emb, cell_mask)
                if include_spatial_cell_emb:
                    spatial_cell_emb = compute_mean_unmasked_emb(n_emb, cell_mask)
                neighborhood_emb = compute_mean_unmasked_emb(n_emb, neighborhood_mask)
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


def harmonize_adata(adata: ad.AnnData,
                    gene_mapping_dict_file_path: str | None='/lustre/scratch126/cellgen/lotfollahi/DATASETS/genes/homo_sapiens_gene_name_to_ensembl_id_dict.pkl',
                    gene_occurrence_count_file_path: str | None='/lustre/scratch126/cellgen/lotfollahi/DATASETS/genes/homo_sapiens_gene_occurence_count_dict.pkl',
                    gene_occurrence_count_filter_value: int=10,
                    ensembl_release: int=111,
                    min_genes_per_cell: int=10,
                    min_cells_per_gene: int=10,
                    ) -> ad.AnnData:
    """
    Harmonize an AnnData object prior to tokenization.

    Parameters
    -----------
    adata:
        An unharmonized AnnData object.
    ensembl_release:
        Ensembl release used to retrieve ensembl IDs.
    min_genes_per_cell:
        Minimum amount of genes per cell for a cell not to be filtered.

    Returns:
    -----------
    adata:
        A harmonized AnnData object.
    """
    print('==================================================')
    print('STEP 1: DATA VALIDATION...')
    print('==================================================')
    print('Checking that adata.X contains raw counts...')
    if issparse(adata.X):
        data = adata.X.data
    else:
        data = np.asarray(adata.X)
    all_integers = np.allclose(data, data.astype(int))

    print('==================================================')
    print('STEP 2: ADDING ENSEMBL IDS...')
    print('==================================================')
    if not gene_mapping_dict_file_path:
        print(f'Adding ensembl IDs from release {ensembl_release}...')
        print(f'Make sure this ensembl release is aligned with pretraining.')
        print(f'Current ensembl release used for pretraining is 111.')
        # Extract ensembl IDs of protein coding and miRNA mouse genes
        ensembl = EnsemblRelease(release=ensembl_release, species='human')
        ensembl.download()
        ensembl.index()
        all_genes = ensembl.genes()
        protein_coding_genes = [
            gene for gene in all_genes if gene.biotype == "protein_coding"]
        mirna_genes = [
            gene for gene in all_genes if gene.biotype == "miRNA"]
        all_relevant_genes = protein_coding_genes + mirna_genes
        gene_ensembl_map_dict = {
            gene.gene_name: gene.gene_id for gene in all_relevant_genes}
    else:
        print(f'Adding ensembl IDs from gene_mapping_dict with file path `{gene_mapping_dict_file_path}`...')
        with open(gene_mapping_dict_file_path, 'rb') as f:
            gene_ensembl_map_dict = pickle.load(f)

    adata_gene_names = [gene_name for gene_name in adata.var_names.tolist()]
    adata.var.index = adata_gene_names
    
    harmonized_gene_names = []
    matching_ensembl_ids = []
    for gene_name in adata_gene_names:
        if gene_name in gene_ensembl_map_dict.keys():
            harmonized_gene_names.append(gene_name)
            matching_ensembl_ids.append(gene_ensembl_map_dict[gene_name])
    print(f'Number of genes with matching ensembl IDs: {len(harmonized_gene_names)}.')
    print(f'Number of genes skipped due to non-matching ensembl IDs: {len(adata_gene_names) - len(harmonized_gene_names)}.')
    if len(adata_gene_names) - len(harmonized_gene_names) > 0:
        print(f'Genes excluded due to non-matching ensembl IDs: {set(adata_gene_names) - set(harmonized_gene_names)}.')

    adata = adata[:, adata.var.index.isin(harmonized_gene_names)].copy()
    adata.var = pd.DataFrame(
        index=pd.Index(harmonized_gene_names, name='gene_name'),
        data={'ensembl_id': matching_ensembl_ids})

    print(f'Filtering genes that have not occurred enough during pretraining...')
    with open(gene_occurrence_count_file_path, 'rb') as f:
        gene_occurrence_count_dict = pickle.load(f)

    all_ensembl_ids = adata.var['ensembl_id'].tolist()
    keep_ensembl_ids = [
        ensembl_id for ensembl_id in all_ensembl_ids if gene_occurrence_count_dict[
                ensembl_id] > gene_occurrence_count_filter_value]

    print(f'Number of genes skipped due to not enough pretraining occurrences: {len(all_ensembl_ids) - len(keep_ensembl_ids)}.')
    if len(all_ensembl_ids) - len(keep_ensembl_ids) > 0:
        print(f'Genes excluded due to not enough pretraining occurrences: {set(all_ensembl_ids) - set(keep_ensembl_ids)}.')
    adata = adata[:, adata.var['ensembl_id'].isin(keep_ensembl_ids)].copy()

    print('==================================================')
    print('STEP 3: BASIC QUALITY CONTROL...')
    print('==================================================')
    # Filter cells with less than min_genes_per_cell genes
    print(f'Filtering cells with less than {min_genes_per_cell} genes.')
    sc.pp.filter_cells(
        adata,
        min_genes=min_genes_per_cell)

    # Filter genes with less than min_cells_per_gene cells
    print(f'Filtering genes expressed in less than {min_cells_per_gene} cells.')
    sc.pp.filter_genes(
        adata,
        min_cells=min_cells_per_gene)

    # Add dummy values as special values
    print('==================================================')
    print('STEP 4: ADDING SPECIAL VALUES...')
    print('==================================================')
    if 'cell_id' not in adata.obs.keys():
        adata.obs['cell_id'] = adata.obs_names
    if 'dataset_id' not in adata.uns.keys():
        adata.uns['dataset_id'] = 14  # just dummy values
    if 'batch' not in adata.uns.keys():
        adata.uns['batch'] = 'batch0' # just dummy values
    if 'assay' not in adata.uns.keys():
        adata.uns['assay'] = 'xenium' # just dummy values
    if 'species' not in adata.uns.keys():
        adata.uns['species'] = 'homo_sapiens' # just dummy values
    if 'tissue' not in adata.uns.keys():
        adata.uns['tissue'] = 'lung' # just dummy values

    return adata


def tokenize_adata(adata: ad.AnnData,
                   model_folder_path: str,
                   cache_directory_path: str,             
                   nproc: int = 4,
                   processing_mode: Literal['sequential',
                                            'parallel'] = 'parallel',
                   add_neigh_cell_ids: bool = False,
                   use_generator: bool = True,
                   keep_in_memory: bool = False,
                   ) -> Dataset:
    """
    Harmonize and tokenize an AnnData object based on the parameters in the
    model config and return the tokenized huggingface dataset and harmonized
    AnnData object.

    Parameters
    -----------
    adata:
        AnnData object to be tokenized.
    model_folder_path:
        Path to the folder containing the model config, token dictionary, and
        normalization factors.
    cache_directory_path:
        Path where the cache is stored during dataset creation.     
    n_proc:
        Number of processes used.
    processing_mode:
        Mode of processing.
    add_neigh_cell_ids:
        Whether neighbor cell IDs should be stored in tokenized data (used for
        perturbations).
    use_generator:
        Whether to use generator for dataset creation.
    keep_in_memory:
        Whether to keep dataset in memory.

    Returns
    -----------
    dataset:
        The tokenized data stored in a huggingface dataset.
    """
    print('==================================================')
    print('STEP 1: LOADING CONFIG...')
    print('==================================================')
    model_config_file_path = Path(model_folder_path) / 'model_config.yaml'
    token_dictionary_file_path = Path(model_folder_path) / 'token_dictionary.pkl'
    norm_factor_file_path = Path(model_folder_path) / 'norm_factors.csv'

    # Load model config
    with open(model_config_file_path, 'r') as file:
        model_config = yaml.safe_load(file)

    print('==================================================')
    print('STEP 2: TOKENIZING ANNDATA OBJECT...')
    print('==================================================')
    # Tokenize adata
    if model_config['data']['tokenizer_type'] == 'cell_neighborhood':
        Tokenizer = cell_tokenizers.CellNeighborhoodTokenizer
    elif model_config['data']['tokenizer_type'] == 'cell_graph':
        Tokenizer = cell_tokenizers.CellGraphTokenizer
    tk = Tokenizer(
        nproc=nproc,
        processing_mode=processing_mode,
        model_input_size=model_config['data']['model_input_size'],
        n_neighs=model_config['data']['n_neighs'],
        radius=None,
        delaunay=False,
        rank_cell_norm_method=model_config['data']['rank_cell_norm_method'],
        rank_gene_norm_method=model_config['data']['rank_gene_norm_method'],
        rank_count_norm_method=model_config['data']['rank_count_norm_method'],
        count_cell_norm_method=model_config['data']['count_cell_norm_method'],
        count_gene_norm_method=model_config['data']['count_gene_norm_method'],
        count_count_norm_method=model_config['data']['count_count_norm_method'],
        norm_factor_file_path=norm_factor_file_path,
        token_dictionary_file_path=token_dictionary_file_path,
        add_neigh_cell_ids=add_neigh_cell_ids)
    dataset_dict = tk._tokenize_adata(adata=adata)
    dataset = tk._create_dataset(
        dataset_dict=dataset_dict,
        use_generator=use_generator,
        cache_directory_path=cache_directory_path,
        keep_in_memory=keep_in_memory)

    columns = list(dataset.features.keys())
    columns.remove("cell_id")
    dataset.set_format(
        type="torch",
        columns=columns,
        output_all_columns=True)
    
    return dataset


@torch.inference_mode()
def embed_dataset(dataset: Dataset,
                  model_folder_path: str,
                  emb_layer: int | None = None,
                  agg_excluded_tokens: list[int] | None = None,
                  top_k: int | None = None,
                  batch_size: int = 128,
                  pin_memory: bool = False,
                  num_workers: int = 12,
                  include_spatial_cell_emb: bool = True,
                  return_token_embeddings: bool = False,
                  ) -> dict:
    """
    Parameters
    -----------
    dataset:
        Tokenized huggingface dataset.
    model_folder_path:
        Path to the folder containing the model config, token dictionary, and
        normalization factors.
    emb_layer:
        Layer for which to retrieve the embedding.
    agg_excluded_tokens:
        List of tokens to be excluded from the aggregation.
    top_k:
        Include only top_k genes in aggregation.
    batch_size:
        Dataloader param.
    pin_memory:
        Dataloader param.
    num_workers:
        Number of workers used.
    include_spatial_cell_emb:
        If `True`, also return a spatially contextualized cell embedding that
        attends to the neighborhood.
    return_token_embeddings:
        If `True`, also return per-token embeddings for each sequence position
        (cell and neighborhood tokens; special tokens are excluded).

    Returns:
    -----------
    output_embed:
        Dictionary with the cell, cell gene, neighborhood, and neighborhood gene
        embeddings.
    """
    print('==================================================')
    print('STEP 1: LOADING CONFIG...')
    print('==================================================')
    model_config_file_path = Path(model_folder_path) / 'model_config.yaml'
    token_dictionary_file_path = Path(model_folder_path) / 'token_dictionary.pkl'
    norm_factor_file_path = Path(model_folder_path) / 'norm_factors.csv'
    model_checkpoint_path = Path(model_folder_path) / 'model_checkpoint.pt'

    # Load model config
    with open(model_config_file_path, 'r') as file:
        model_config = yaml.safe_load(file)

    # Get token sequence length and number of special tokens
    n_special_tokens = len(model_config['meta']['special_tokens'])
    seq_len = (
        model_config['data']['seq_len_cell'] +
        model_config['data']['seq_len_neighborhood'] +
        n_special_tokens)

    # Specify last emb layer if not defined
    if emb_layer is None:
        emb_layer = model_config['meta']['enc_depth'] 

    # Load token dict and get token dict-specfic params
    with open(token_dictionary_file_path, 'rb') as file:
        token_dict = pickle.load(file)
    vocab_size = len(token_dict)
    n_special_values = sum(1 for key in token_dict if "spv" in key)

    print('==================================================')
    print('STEP 2: GENERATING EMBEDDINGS...')
    print('==================================================')
    # Set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # Initialize encoder, predictor, and target encoder
    target_encoder, _ = init_model(
        gt_type=model_config['meta']['gt_type'],
        count_encoding=model_config['meta']['count_encoding'],
        n_value_bins=model_config['meta']['n_value_bins'],
        cell_pos_enc=model_config['meta']['cell_pos_enc'],
        device=device,
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_special_tokens=n_special_tokens,
        n_segments=model_config['data']['n_segments'],
        n_special_values=n_special_values,
        enc_emb_dim=model_config['meta']['enc_emb_dim'],
        enc_depth=model_config['meta']['enc_depth'],
        pred_emb_dim=model_config['meta']['pred_emb_dim'],
        pred_depth=model_config['meta']['pred_depth'],
        num_heads=model_config['meta']['num_heads'],
        mlp_ratio=model_config['meta']['mlp_ratio'],
        use_flash_attention=model_config['meta']['use_flash_attention'],
        api_version=model_config['meta']['api_version'],
        sep_gene_tokens_neb=model_config['data']['sep_gene_tokens_neb'],
        predict_gene=model_config['meta']['predict_gene'],
        pos_learnable=model_config['meta']['pos_learnable'])

    if model_config['meta']['api_version'] != 'v3':
        return_layer_emb_fn = target_encoder.return_layer_emb
    else:
        return_layer_emb_fn = target_encoder.backbone.return_layer_emb

    # Create mask collator
    mask_collator = BlockMaskCollator(
        n_targets=model_config['mask']['n_targets'],
        n_contexts=model_config['mask']['n_contexts'],
        n_segments=model_config['data']['n_segments'],
        seq_len_cell=model_config['data']['seq_len_cell'],
        seq_len_neighborhood=model_config['data']['seq_len_neighborhood'],
        n_special_tokens=n_special_tokens,
        per_block_mask_ratio=model_config['mask']['per_block_mask_ratio'],
        sample_segments=False,
        sample_gene_masks=False)
        
    # Create torch dataset
    cell_dataset = init_cell_dataset(
        dataset=dataset,
        vocab_size=vocab_size,
        seq_len_cell=model_config['data']['seq_len_cell'],
        seq_len_neighborhood=model_config['data']['seq_len_neighborhood'],
        tokenizer_type=model_config['data']['tokenizer_type'],
        gt_type=model_config['meta']['gt_type'],
        cell_pos_enc=model_config['meta']['cell_pos_enc'],
        special_tokens=model_config['meta']['special_tokens'],
        sampling_strategy=None,
        n_nonzero_tokens_list=[],
        include_cell_id=True,
        sep_gene_tokens_neb=model_config['data']['sep_gene_tokens_neb'])

    # Initialize dataloader
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
        persistent_workers=False)

    # Load model checkpoint
    _, _, target_encoder, _, _, start_epoch, _ = load_checkpoint(
            device=device,
            r_path=model_checkpoint_path,
            encoder=None,
            predictor=None,
            target_encoder=target_encoder,
            opt=None,
            scaler=None,
            is_training=False)
    target_encoder.eval()

    # Retrieve embeddings
    all_cell_emb_list = []
    if include_spatial_cell_emb:
        all_spatial_cell_emb_list = []
    all_neighborhood_emb_list = []
    if return_token_embeddings:
        all_token_emb_list = []

    for itr, (udata, _, _, masks_attention) in tqdm(enumerate(loader)):
        for key in udata.keys():
            if key != 'cell_id':
                udata[key] = udata[key].to(device, non_blocking=True)
        masks_attention = masks_attention.to(device, non_blocking=True)

        # Aggregate gene embeddings into cell and neighborhood embeddings
        ns_tokens = udata['tokens'][:, n_special_tokens:]

        # Retrieve gene embeddings from different layers
        with torch.cuda.amp.autocast(
            dtype=torch.bfloat16,
            enabled=model_config['meta']['use_bfloat16']):

            emb_layers = [emb_layer]

            full_ctx, cell_only_ctx = return_layer_emb_fn(
                layers=emb_layers,
                batch=udata,
                masks_attention=masks_attention,
                need_cell_only_context=True,
            )

            c_emb = cell_only_ctx[emb_layer].cpu()
            n_emb = full_ctx[emb_layer].cpu()
            if return_token_embeddings:
                # n_emb contains embeddings for all tokens
                all_token_emb_list.append(n_emb)
        
        # Create mask for index cell genes
        cell_mask = create_binary_selection_mask(
            ns_tokens,
            selection_type="agg_cell",
            excluded_tokens=agg_excluded_tokens,
            seq_len_cell=model_config['data']['seq_len_cell'],
            top_k=top_k).cpu()

        # Create mask for neighbor cell genes
        if model_config['data']['tokenizer_type'] == 'cell_neighborhood':
            neighborhood_mask = create_binary_selection_mask(
                ns_tokens,
                selection_type="agg_neighborhood",
                excluded_tokens=agg_excluded_tokens,
                seq_len_cell=model_config['data']['seq_len_cell'],
                top_k=top_k).cpu()
        elif model_config['data']['tokenizer_type'] == 'cell_graph':
            neighborhood_mask = create_binary_selection_mask(
                ns_tokens,
                selection_type="agg_graph",
                excluded_tokens=agg_excluded_tokens,
                seq_len_cell=model_config['data']['seq_len_cell'],
                top_k=top_k,
                n_segments=model_config['data']['n_segments']).cpu()

        # Average gene embeddings into cell and neighborhood embedding                    
        cell_emb = compute_mean_unmasked_emb(c_emb, cell_mask)
        if include_spatial_cell_emb:
            spatial_cell_emb = compute_mean_unmasked_emb(n_emb, cell_mask)
        neighborhood_emb = compute_mean_unmasked_emb(n_emb, neighborhood_mask)

        all_cell_emb_list.append(cell_emb)
        if include_spatial_cell_emb:
            all_spatial_cell_emb_list.append(spatial_cell_emb)
        all_neighborhood_emb_list.append(neighborhood_emb)

    output_embed = {}        

    # Store cell, spatially contextualized cell and neighborhood embeddings of
    # all observations
    output_embed["cell_emb"] = np.array(torch.cat(
        all_cell_emb_list,
        dim=0))
    output_embed["neighborhood_emb"] = np.array(torch.cat(
        all_neighborhood_emb_list,
        dim=0))
    if include_spatial_cell_emb:
        output_embed["spatial_cell_emb"] = np.array(torch.cat(
            all_spatial_cell_emb_list,
            dim=0))        
    if return_token_embeddings:
        output_embed["token_emb"] = np.array(torch.cat(
            all_token_emb_list,
            dim=0))
    return output_embed


def _build_perturb_index(df: pd.DataFrame) -> Dict[str, List[dict]]:
    """
    Turn perturb_df into an index:
        {perturbed_cell_id: [row_as_dict, …]}
    (Using itertuples avoids Python object creation for every column.)
    """
    index = defaultdict(list)
    for row in df.itertuples(index=False):
        index[row.perturbed_cell_id].append(row._asdict())
    return index

def _perturb_batch(
    batch: dict,
    index: Dict[str, List[dict]],
    seq_len_cell: int = 256,
) -> dict:
    """
    `batch` is the dictionary of column -> list-of-values
    returned by Hugging-Face when batched=True.
    Modify the tensors in-place (cheap) and return the same dict.
    """
    B = len(batch["cell_ids"])          # batch size
    for b in range(B):
        cell_ids: List[str] = list(dict.fromkeys(batch["cell_ids"][b]))
        # Fast reject: does this example touch any perturbed cell at all?
        if not any(cid in index for cid in cell_ids):
            continue

        # Handle index-cell (first) and neighbourhood (rest) separately
        for cid in cell_ids:
            for row in index.get(cid, []):
                is_index_cell = (cid == cell_ids[0])
                if row["perturbation_target"] == "cell" and not is_index_cell:
                    continue
                if row["perturbation_target"] == "neighborhood" and is_index_cell:
                    continue

                gene_tokens = batch["gene_tokens"][b]
                gene_expr   = batch["gene_expr"][b]

                # --- choose which token positions to perturb -------------
                if row["perturbed_gene_token"] == "all":
                    if is_index_cell:
                        idx = slice(0, seq_len_cell)
                    else:
                        idx = slice(seq_len_cell, None)
                else:
                    token_id = row["perturbed_gene_token"]
                    token_slice = (
                        gene_tokens[:seq_len_cell]
                        if is_index_cell else
                        gene_tokens[seq_len_cell:]
                    )
                    rel_idx = torch.nonzero(token_slice == token_id, as_tuple=True)[0]
                    offset  = 0 if is_index_cell else seq_len_cell
                    idx     = rel_idx + offset
                # ---------------------------------------------------------

                if row["perturbation_type"] == "knockout":
                    gene_expr[idx]   = 0.0
                elif row["perturbation_type"] == "foldchange":
                    gene_expr[idx] *= row["foldchange"]
                else:
                    raise ValueError(f"Bad perturbation_type: {row['perturbation_type']}")

    return batch

def perturb_dataset(
    dataset: Dataset,
    perturb_df: pd.DataFrame,
    model_folder_path: str,
    seq_len_cell: int = 256,
    nproc: int = 4,
    batch_size: int = 1000,
    keep_in_memory: bool = False,
) -> Dataset:
    # 1) Load token dictionary once
    with open(Path(model_folder_path) / "token_dictionary.pkl", "rb") as f:
        token_dict = pickle.load(f)

    # 2) Convert gene IDs to token IDs up-front (vectorised)
    perturb_df = perturb_df.copy()
    perturb_df["perturbed_gene_token"] = perturb_df["perturbed_ensembl_id"].where(
        perturb_df["perturbed_ensembl_id"] == "all",
        perturb_df["perturbed_ensembl_id"].map(token_dict),
    )

    # 3) Build an index for O(1) lookup
    perturb_index = _build_perturb_index(perturb_df)
    # 4) Partial-apply so the mapper sees only one argument
    perturb_fn = partial(
        _perturb_batch,
        index=perturb_index,
        seq_len_cell=seq_len_cell,
    )
    # 5) Map in batch mode
    return dataset.map(
        perturb_fn,
        batched=True,
        batch_size=batch_size,
        num_proc=nproc,
        keep_in_memory=keep_in_memory,
        load_from_cache_file=False,
    )


@torch.inference_mode()
def harmonize_tokenize_embed_pipeline(
        adata: ad.AnnData,
        sample_key: str | None,
        model_folder_path: str,
        gene_perturb_df: pd.DataFrame | None = None,               
        nproc: int = 4,
        processing_mode: Literal['sequential',
                                 'parallel'] = 'parallel',
        save_dataset_path: Path | str | None = None,
        num_shards: int = 32,
        emb_layer: int | None = None,
        agg_excluded_tokens: list[int] | None = None,
        top_k: int | None = None,
        batch_size: int = 128,
        pin_memory: bool = False,
        num_workers: int = 12
        ) -> ad.AnnData:
    """
    Harmonize, tokenize and embed an AnnData object.

    Parameters
    -----------
    adata:
        An unharmonized AnnData object to be tokenized.
    sample_key:
        Key in `adata.obs` where the sample information is stored.
    model_folder_path:
        Path to the folder containing the model config, token dictionary, and
        normalization factors.
    gene_perturb_df:
        DataFrame with perturbation data, e.g.
        ```
        gene_perturb_df =  pd.DataFrame({
            'ensembl_id': ['ENSG00000169194', 'ENSG00000131724'],
            'target': ['neighborhood', 'cell'],
            'perturbation_type': ['foldchange', 'knockout'],
            'foldchange': [0.5, np.nan]
        })
        ```.
    n_proc:
        Number of processes used for tokenization.
    processing_mode:
        Mode of processing used for tokenization.
    save_dataset_path:
        If specified, huggingface dataset is written to disk at this path.
    num_shards:
        Number of shards with which huggingface dataset is saved.
    emb_layer:
        Layer for which to retrieve the embedding.
    agg_excluded_tokens:
        List of tokens to be excluded from the aggregation.
    top_k:
        Include only top_k genes in aggregation.
    batch_size:
        Dataloader param.
    pin_memory:
        Dataloader param.
    num_workers:
        Number of workers used for model inference.

    Returns:
    -----------
    adata:
        A harmonized AnnData object with embeddings stored in `adata.obsm`.
    """
    datasets = []
    if sample_key:
        samples = adata.obs[sample_key].unique().tolist()

        for sample in samples:
            adata_sample = adata[adata.obs[sample_key] == sample]
            print(f"Start processing sample: {sample}...")
            print(f"Harmonizing sample {sample}...")
            adata_sample = harmonize_adata(adata_sample)
            print(f"Harmonized sample {sample}.")

            print(f"Tokenizing sample {sample}...")
            dataset_sample = tokenize_adata(
                adata=adata_sample,
                model_folder_path=model_folder_path,             
                nproc=nproc,
                processing_mode=processing_mode)
            datasets.append(dataset_sample)
            print(f"Tokenized sample {sample}.")

        print(f"Concatenating tokenized data...")
        dataset = concatenate_datasets(datasets)
        print(f"Concatenated tokenized data.")

    else:
        print("No `sample_key` specified. Start processing entire AnnData.")
        print(f"Harmonizing AnnData...")
        adata = harmonize_adata(adata)
        print(f"Harmonized AnnData.")

        print(f"Tokenizing AnnData.")
        dataset = tokenize_adata(
            adata=adata,
            model_folder_path=model_folder_path,             
            nproc=nproc,
            processing_mode=processing_mode)        

    if save_dataset_path:
        print(f"Saving tokenized data...")
        dataset.save_to_disk(
            save_dataset_path,
            num_shards=num_shards)
        print(f"Saved tokenized data.")

    print(f"Embedding tokenized data...")
    output_embed = embed_dataset(
        dataset=dataset,
        model_folder_path=model_folder_path,
        emb_layer=emb_layer,
        agg_excluded_tokens=agg_excluded_tokens,
        top_k=top_k,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers)
    print(f"Embedded tokenized data.")

    # Add embeddings to adata
    for key, values in output_embed.items():
        adata.obsm[key] = values

    return adata


@torch.inference_mode()
def gene_embed_dataset(dataset: Dataset,
                  model_folder_path: str,
                  emb_layer: int | None = None,
                  cell_gene_ids: list = [],
                  neighborhood_gene_ids: list = [],
                  batch_size: int = 128,
                  pin_memory: bool = False,
                  num_workers: int = 12,
                  return_gene: bool=False,
                  return_gene_per_data: bool=False,
                  compute_cosine_with_list:  list[str] = [],
                  return_distance: bool=False,
                  return_cosine_sim: bool=False,
                  return_receptor_average: bool=False,
                  include_spatial_cell_emb: bool = True,
                  description: str='',
                  ) -> dict:
    
    """
    Parameters
    -----------
    dataset:
        Tokenized huggingface dataset.
    model_folder_path:
        Path to the folder containing the model config, token dictionary, and
        normalization factors.
    emb_layer:
        Layer for which to retrieve the embedding.
    cell_gene_ids:
        List with gene IDs for which cell gene embeddings will be retrieved.
    neighborhood_gene_ids:
        List with gene IDs for which neighborhood gene embeddings will be
        retrived.
    batch_size:
        Dataloader param.
    pin_memory:
        Dataloader param.
    num_workers:
        Number of workers used.
    return_gene_per_data:
        If 'True' will return gene_embedding for each gene per dataset.
    compute_cosine_with_list:
       A list that defines the items with which we want to compute cosine similarity.
       it could have value of 'cell' or/and 'neighborhood'.
    return_distance:
        If 'True' will compute and return distance between cosine sim of cell_neb 
        and cell_cell matrix.
    return_cosine_sim: 
        If 'True' will compute and return cosine_sim matrix.
    return_receptor_average:
        If 'True' will compute and return receptor average embeddings for cell-neighborhood gene pairs.
    include_spatial_cell_emb:
        If `True`, also return gene embeddings for spatially contextualized cell embedding that
        attends to the neighborhood.
    description:
        description for task that is currently using this function.
    Returns:
    -----------
    output_embed:
        Dictionary with the cell, cell gene, neighborhood, and neighborhood gene
        embeddings.
    """
    print('==================================================')
    print('STEP 1: LOADING CONFIG...')
    print('==================================================')
    model_config_file_path = Path(model_folder_path) / 'model_config.yaml'
    token_dictionary_file_path = Path(model_folder_path) / 'token_dictionary.pkl'
    norm_factor_file_path = Path(model_folder_path) / 'norm_factors.csv'
    model_checkpoint_path = Path(model_folder_path) / 'model_checkpoint.pt'

    # Load model config
    with open(model_config_file_path, 'r') as file:
        model_config = yaml.safe_load(file)

    # Get token sequence length and number of special tokens
    n_special_tokens = len(model_config['meta']['special_tokens'])
    seq_len = (
        model_config['data']['seq_len_cell'] +
        model_config['data']['seq_len_neighborhood'] +
        n_special_tokens)
    seq_len_cell = model_config['data']['seq_len_cell']
    # Specify last emb layer if not defined
    if emb_layer is None:
        emb_layer = model_config['meta']['enc_depth'] 

    # Load token dict and get token dict-specfic params
    with open(token_dictionary_file_path, 'rb') as file:
        token_dict = pickle.load(file)
    vocab_size = len(token_dict)
    n_special_values = sum(1 for key in token_dict if "spv" in key)

    print('==================================================')
    print(f'STEP 2: {description}')
    print('==================================================')
    # Set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # Initialize encoder, predictor, and target encoder
    target_encoder, _ = init_model(
        gt_type=model_config['meta']['gt_type'],
        count_encoding=model_config['meta']['count_encoding'],
        n_value_bins=model_config['meta']['n_value_bins'],
        cell_pos_enc=model_config['meta']['cell_pos_enc'],
        device=device,
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_special_tokens=n_special_tokens,
        n_segments=model_config['data']['n_segments'],
        n_special_values=n_special_values,
        enc_emb_dim=model_config['meta']['enc_emb_dim'],
        enc_depth=model_config['meta']['enc_depth'],
        pred_emb_dim=model_config['meta']['pred_emb_dim'],
        pred_depth=model_config['meta']['pred_depth'],
        num_heads=model_config['meta']['num_heads'],
        mlp_ratio=model_config['meta']['mlp_ratio'],
        use_flash_attention=model_config['meta']['use_flash_attention'],
        api_version=model_config['meta']['api_version'],
        sep_gene_tokens_neb=model_config['data']['sep_gene_tokens_neb'])

    if model_config['meta']['api_version'] != 'v3':
        return_layer_emb_fn = target_encoder.return_layer_emb
    else:
        return_layer_emb_fn = target_encoder.backbone.return_layer_emb

    # Create mask collator
    mask_collator = BlockMaskCollator(
        n_targets=model_config['mask']['n_targets'],
        n_contexts=model_config['mask']['n_contexts'],
        n_segments=model_config['data']['n_segments'],
        seq_len_cell=model_config['data']['seq_len_cell'],
        seq_len_neighborhood=model_config['data']['seq_len_neighborhood'],
        n_special_tokens=n_special_tokens,
        per_block_mask_ratio=model_config['mask']['per_block_mask_ratio'],
        sample_segments=False,
        sample_gene_masks=False)
        
    # Create torch dataset
    cell_dataset = init_cell_dataset(
        dataset=dataset,
        vocab_size=vocab_size,
        seq_len_cell=model_config['data']['seq_len_cell'],
        seq_len_neighborhood=model_config['data']['seq_len_neighborhood'],
        tokenizer_type=model_config['data']['tokenizer_type'],
        gt_type=model_config['meta']['gt_type'],
        cell_pos_enc=model_config['meta']['cell_pos_enc'],
        special_tokens=model_config['meta']['special_tokens'],
        sampling_strategy=None,
        n_nonzero_tokens_list=[],
        include_cell_id=True,
        sep_gene_tokens_neb=model_config['data']['sep_gene_tokens_neb'])

    # Initialize dataloader
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
        persistent_workers=False)

    # Load model checkpoint
    _, _, target_encoder, _, _, start_epoch, _ = load_checkpoint(
            device=device,
            r_path=model_checkpoint_path,
            encoder=None,
            predictor=None,
            target_encoder=target_encoder,
            opt=None,
            scaler=None,
            is_training=False)
    target_encoder.eval()

    # Retrieve embeddings
    all_cell_gene_emb_dict = {}
    all_neighborhood_gene_emb_dict = {}
    all_cell_gene_emb_per_data_dict = {}
    all_neighborhood_gene_emb_per_data_dict = {}
    if include_spatial_cell_emb:
        all_spatial_cell_gene_emb_dict = {}
        all_spatial_cell_gene_emb_per_data_dict = {}
    cos_sim_dict = {}
    emd_list = []
    emd_matrix_list = []
    receptor_average_dict = {}
    MAX_OCC = model_config['data']['n_segments'] -1 


    for itr, (udata, _, _, masks_attention) in tqdm(enumerate(loader)):
        for key in udata.keys():
            if key != 'cell_id':
                udata[key] = udata[key].to(device, non_blocking=True)
        masks_attention = masks_attention.to(device, non_blocking=True)

        # Aggregate gene embeddings into cell and neighborhood embeddings
        ns_tokens = udata['tokens'][:, n_special_tokens:]

        # Retrieve gene embeddings from different layers
        with torch.cuda.amp.autocast(
            dtype=torch.bfloat16,
            enabled=model_config['meta']['use_bfloat16']):

            emb_layers = [emb_layer]

            full_ctx, cell_only_ctx = return_layer_emb_fn(
                layers=emb_layers,
                batch=udata,
                masks_attention=masks_attention,
                need_cell_only_context=True,
            )

            # Keep embeddings on device for vectorized ops; move to CPU only when storing
            c_emb = cell_only_ctx[emb_layer]  # (N, total_seq_len, D) cell-only context embeddings
            n_emb = full_ctx[emb_layer]       # (N, total_seq_len, D) full context embeddings (attends neighborhood)
        emb = c_emb
        if len(cell_gene_ids) != 0 or len(neighborhood_gene_ids) != 0:
            N, L, D = emb.shape  # batch size, sequence length, embedding dim
            Gc = len(cell_gene_ids)  # number of requested cell genes
            Gn = len(neighborhood_gene_ids)  # number of requested neighborhood genes

            # Vectorized extraction for cell genes (first occurrence in cell segment)
            if Gc > 0:
                ns_cell = ns_tokens[:, :seq_len_cell]  # (N, Lc) token IDs in the cell segment
                cell_gene_ids_tensor = torch.as_tensor(cell_gene_ids, device=ns_tokens.device, dtype=ns_tokens.dtype)  # (Gc,)
                # Build equality mask across all cell genes and positions: True where token equals gene ID
                eq_cell = (ns_cell.unsqueeze(1) == cell_gene_ids_tensor.view(1, -1, 1))  # (N, Gc, Lc)
                # Presence for each gene in each sequence (1 if present)
                cell_presence = eq_cell.any(dim=2).float()  # (N, Gc)
                # Compute first occurrence index by assigning unmatched positions a large sentinel and taking min
                pos_cell = torch.arange(seq_len_cell, device=ns_tokens.device).view(1, 1, -1)  # (1,1,Lc)
                masked_pos = torch.where(eq_cell, pos_cell, torch.full_like(pos_cell, seq_len_cell))  # (N,Gc,Lc)
                first_idx = masked_pos.min(dim=2).values  # (N, Gc), equals sentinel if absent
                # Gather embeddings at first occurrence from both contexts; clamp handles absent sentinel safely
                gather_idx = first_idx.clamp(max=L - 1)  # (N, Gc)
                cell_embs = torch.gather(emb, 1, gather_idx.unsqueeze(-1).expand(-1, -1, D))  # (N, Gc, D)
                cell_embs = cell_embs * cell_presence.unsqueeze(-1)  # zero-out entries where gene absent
                if include_spatial_cell_emb:
                    spatial_cell_embs = torch.gather(n_emb, 1, gather_idx.unsqueeze(-1).expand(-1, -1, D))  # (N,Gc,D)
                    spatial_cell_embs = spatial_cell_embs * cell_presence.unsqueeze(-1)  # zero-out when absent

                # Return per-gene embeddings
                if return_gene:
                    for j, gene_id in enumerate(cell_gene_ids):  # store per-gene batch embeddings for later concatenation
                        if itr == 0:
                            all_cell_gene_emb_dict[gene_id] = [cell_embs[:, j, :].detach().cpu()]
                            if include_spatial_cell_emb:
                                all_spatial_cell_gene_emb_dict[gene_id] = [spatial_cell_embs[:, j, :].detach().cpu()]
                        else:
                            all_cell_gene_emb_dict[gene_id].append(cell_embs[:, j, :].detach().cpu())
                            if include_spatial_cell_emb:
                                all_spatial_cell_gene_emb_dict[gene_id].append(spatial_cell_embs[:, j, :].detach().cpu())

                # Accumulate per-data averages (sum and count)
                if return_gene_per_data:
                    sum_per_gene = cell_embs.sum(dim=0)  # (Gc, D) sum over batch for each gene
                    count_per_gene = cell_presence.sum(dim=0)  # (Gc,) number of cells where gene present
                    for j, gene_id in enumerate(cell_gene_ids):
                        gene_sum = sum_per_gene[j]
                        gene_count = count_per_gene[j]
                        if itr == 0:
                            all_cell_gene_emb_per_data_dict[gene_id] = (gene_sum.detach().cpu(), gene_count.detach().cpu())
                            if include_spatial_cell_emb:
                                spatial_sum_per_gene = spatial_cell_embs.sum(dim=0)  # (Gc, D)
                                spatial_count_per_gene = count_per_gene           # (Gc,)
                                all_spatial_cell_gene_emb_per_data_dict[gene_id] = (
                                    spatial_sum_per_gene[j].detach().cpu(), spatial_count_per_gene[j].detach().cpu()
                                )
                        else:
                            all_cell_gene_emb_per_data_dict[gene_id][0].add_(gene_sum.detach().cpu())
                            all_cell_gene_emb_per_data_dict[gene_id][1].add_(gene_count.detach().cpu())
                            if include_spatial_cell_emb:
                                spatial_sum_per_gene = spatial_cell_embs.sum(dim=0)
                                spatial_count_per_gene = count_per_gene
                                all_spatial_cell_gene_emb_per_data_dict[gene_id][0].add_(spatial_sum_per_gene[j].detach().cpu())
                                all_spatial_cell_gene_emb_per_data_dict[gene_id][1].add_(spatial_count_per_gene[j].detach().cpu())

            # Compute receptor averages for cell-neighborhood gene pairs (vectorized)
            if return_receptor_average and Gc > 0 and Gn > 0:
                ns_neb = ns_tokens[:, seq_len_cell:]  # (N, L_neb) neighborhood segment tokens
                neigh_gene_ids_tensor = torch.as_tensor(neighborhood_gene_ids, device=ns_tokens.device, dtype=ns_tokens.dtype)  # (Gn,)
                # Equality mask across neighborhood genes
                eq_neb = (ns_neb.unsqueeze(1) == neigh_gene_ids_tensor.view(1, -1, 1))  # (N, Gn, L_neb)
                neigh_presence = eq_neb.any(dim=2).float()  # (N, Gn) presence per sequence/gene

                # Build masks: both present vs cell present and neighborhood absent
                both_mask = (cell_presence.unsqueeze(2) * neigh_presence.unsqueeze(1))  # (N, Gc, Gn)
                absent_mask = (cell_presence.unsqueeze(2) * (1.0 - neigh_presence.unsqueeze(1)))  # (N, Gc, Gn)

                # Accumulate sums and counts for each (cell_gene, neighborhood_gene)
                present_sum = (cell_embs.unsqueeze(2) * both_mask.unsqueeze(-1)).sum(dim=0)  # (Gc, Gn, D)
                present_count = both_mask.sum(dim=0)  # (Gc, Gn)
                absent_sum = (cell_embs.unsqueeze(2) * absent_mask.unsqueeze(-1)).sum(dim=0)  # (Gc, Gn, D)
                absent_count = absent_mask.sum(dim=0)  # (Gc, Gn)

                # Update dictionary with sum and count per pair, keeping CPU tensors
                for ci, cell_gene_id in enumerate(cell_gene_ids):
                    for ni, neigh_gene_id in enumerate(neighborhood_gene_ids):
                        if present_count[ci, ni] > 0:
                            key_present = (cell_gene_id, neigh_gene_id, 'present')
                            if key_present not in receptor_average_dict:
                                receptor_average_dict[key_present] = [present_sum[ci, ni].detach().cpu().clone(), int(present_count[ci, ni].item())]
                            else:
                                receptor_average_dict[key_present][0] += present_sum[ci, ni].detach().cpu()
                                receptor_average_dict[key_present][1] += int(present_count[ci, ni].item())
                        if absent_count[ci, ni] > 0:
                            key_absent = (cell_gene_id, neigh_gene_id, 'absent')
                            if key_absent not in receptor_average_dict:
                                receptor_average_dict[key_absent] = [absent_sum[ci, ni].detach().cpu().clone(), int(absent_count[ci, ni].item())]
                            else:
                                receptor_average_dict[key_absent][0] += absent_sum[ci, ni].detach().cpu()
                                receptor_average_dict[key_absent][1] += int(absent_count[ci, ni].item())

            # Process neighborhood genes (multiple occurrences) in a vectorized way
            neb_occ_dict = {}
            if Gn > 0:
                neigh_gene_ids_tensor = torch.as_tensor(neighborhood_gene_ids, device=ns_tokens.device, dtype=ns_tokens.dtype)
                for compute_cosine_with in compute_cosine_with_list:
                    emb_ctx = n_emb if compute_cosine_with == 'neighborhood' else c_emb  # select embedding context
                    if compute_cosine_with == 'neighborhood':
                        seg = ns_tokens[:, seq_len_cell:]  # (N, L_neb) neighborhood tokens
                        base_offset = seq_len_cell        # offset indices into full sequence positions
                    else:
                        seg = ns_tokens[:, :seq_len_cell]  # (N, L_cell) cell tokens
                        base_offset = 0                     # no offset in cell segment
                    Lseg = seg.shape[1]
                    # Equality mask across all genes: True where token equals the gene id
                    eq_all = (seg.unsqueeze(1) == neigh_gene_ids_tensor.view(1, -1, 1))  # (N, Gn, Lseg)
                    occ_counts = eq_all.sum(dim=2)  # (N, Gn) number of occurrences per gene per sample
                    pos = torch.arange(Lseg, device=seg.device).view(1, 1, -1)  # (1,1,Lseg) absolute positions
                    # Replace non-matches with sentinel Lseg and sort to bring valid positions first
                    indices_all = torch.where(eq_all, pos, torch.full_like(pos, Lseg))  # (N,Gn,Lseg)
                    sorted_pos, _ = indices_all.sort(dim=2)  # (N,Gn,Lseg) ascending order
                    K = MAX_OCC  # always return up to MAX_OCC occurrences to match previous API
                    occ_indices = sorted_pos[:, :, :K] + base_offset  # (N, Gn, K) positions into full sequence
                    # Clamp indices to valid range; invalid positions are masked out later
                    occ_indices = occ_indices.clamp_max(L - 1)
                    range_k = torch.arange(K, device=seg.device).view(1, 1, -1)  # (1,1,K)
                    occ_mask = (range_k < occ_counts.unsqueeze(-1)).float()  # (N, Gn, K) 1 for valid, 0 for padded
                    # Gather embeddings along sequence dimension with expanded gene axis
                    emb_ctx_exp = emb_ctx.unsqueeze(1).expand(-1, Gn, -1, -1)  # (N, Gn, L, D)
                    gene_occ = torch.gather(emb_ctx_exp, 2, occ_indices.unsqueeze(-1).expand(-1, -1, -1, D))  # (N, Gn, K, D)

                    # Optionally store per-gene mean embeddings for 'neighborhood'
                    if return_gene and compute_cosine_with == 'neighborhood':
                        sum_occ = (gene_occ * occ_mask.unsqueeze(-1)).sum(dim=2)         # (N, Gn, D) sum over occurrences
                        cnt_occ = occ_mask.sum(dim=2).unsqueeze(-1)                       # (N, Gn, 1) count per sample/gene
                        mean_emb = sum_occ / (cnt_occ + 1e-9)                              # (N, Gn, D) mean over valid occs
                        for j, gene_id in enumerate(neighborhood_gene_ids):
                            if itr == 0:
                                all_neighborhood_gene_emb_dict[gene_id] = [mean_emb[:, j, :].detach().cpu()]
                            else:
                                all_neighborhood_gene_emb_dict[gene_id].append(mean_emb[:, j, :].detach().cpu())

                    if return_gene_per_data and compute_cosine_with == 'neighborhood':
                        # Compute per-sample mean embedding for each gene first
                        sum_occ = (gene_occ * occ_mask.unsqueeze(-1)).sum(dim=2)         # (N, Gn, D)
                        cnt_occ = occ_mask.sum(dim=2).unsqueeze(-1)                      # (N, Gn, 1)
                        mean_emb = sum_occ / (cnt_occ + 1e-9)                             # (N, Gn, D)
                        # For each gene, accumulate sum of non-zero rows and count of non-zero rows across batch
                        for j, gene_id in enumerate(neighborhood_gene_ids):
                            gene_sum, gene_count = compute_sum_and_nonzero_count(mean_emb[:, j, :].detach().cpu())
                            if itr == 0:
                                all_neighborhood_gene_emb_per_data_dict[gene_id] = (gene_sum, gene_count)
                            else:
                                all_neighborhood_gene_emb_per_data_dict[gene_id][0].add_(gene_sum)
                                all_neighborhood_gene_emb_per_data_dict[gene_id][1].add_(gene_count)

                    # For cosine/distance, keep stacked occurrence tensors
                    if len(neighborhood_gene_ids) != 0 and (return_cosine_sim or return_distance):
                        neb_occ_dict[compute_cosine_with] = (gene_occ, occ_mask)

            # Compute cosine similarity components using our function for multiple occurrences.
            if return_cosine_sim:
                for compute_cosine_with in compute_cosine_with_list: 
                    if itr == 0:
                        s, p, c = compute_count_mean_cosine_sim(cell_embs,
                                                                 cell_presence, 
                                                                 neb_occ_dict[compute_cosine_with][0], 
                                                                 neb_occ_dict[compute_cosine_with][1])
                        cos_sim_dict[compute_cosine_with] = (
                            s.detach().cpu(), p.detach().cpu(), c.detach().cpu()  # move to CPU for numpy compatibility
                        )
                    else:
                        sum_cos_sim_temp, pair_count_temp, cell_count_temp = compute_count_mean_cosine_sim(cell_embs,
                                                                                                           cell_presence,                                
                                                                                                           neb_occ_dict[compute_cosine_with][0],  
                                                                                                           neb_occ_dict[compute_cosine_with][1])
                        sum_cos_sim, pair_count, cell_count = cos_sim_dict[compute_cosine_with]
                        cos_sim_dict[compute_cosine_with] = (
                                sum_cos_sim + sum_cos_sim_temp.detach().cpu(),   # accumulate on CPU
                                pair_count + pair_count_temp.detach().cpu(),
                                cell_count + cell_count_temp.detach().cpu()
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
                    cos_sim_temp.append((sum_cos_sim.detach().cpu())/(pair_count.detach().cpu()))  # per-sample matrices on CPU
                _, emd_out, emd_matrix = batch_rowwise_distances(cos_sim_temp[0], cos_sim_temp[1])
                emd_list.append(emd_out)
                emd_matrix_list.append(emd_matrix)
    # last layer
    if return_gene:
        if include_spatial_cell_emb:
            return all_cell_gene_emb_dict, all_neighborhood_gene_emb_dict, all_spatial_cell_gene_emb_dict
        else:
            return all_cell_gene_emb_dict, all_neighborhood_gene_emb_dict
    if return_gene_per_data:
        if include_spatial_cell_emb:
            return all_cell_gene_emb_per_data_dict, all_neighborhood_gene_emb_per_data_dict, all_spatial_cell_gene_emb_per_data_dict
        else:
            return all_cell_gene_emb_per_data_dict, all_neighborhood_gene_emb_per_data_dict
    if return_cosine_sim:
        return cos_sim_dict
    if return_distance:
        return emd_list, emd_matrix_list
    if return_receptor_average:
        return receptor_average_dict

@torch.inference_mode()
def get_gene_embed(
    dataset: Dataset,
    model_folder_path: str,
    emb_layer: int | None = None,
    cell_gene_ensembl_id: list = [],
    neighborhood_gene_ensembl_id: list = [],
    batch_size: int = 128,
    pin_memory: bool = False,
    num_workers: int = 12,
    include_spatial_cell_emb: bool = True,
) -> dict[str, np.ndarray]:
    """
    Retrieve gene embeddings for specified cell and neighborhood gene IDs.

    Parameters
    -----------
    dataset: Tokenized huggingface dataset.
    model_folder_path: Path to the folder containing the model config, token dictionary, and normalization factors.
    emb_layer: Layer for which to retrieve the embedding.
    cell_gene_ensembl_id: List with gene IDs for which cell gene embeddings will be retrieved.
    neighborhood_gene_ensembl_id: List with gene IDs for which neighborhood gene embeddings will be retrieved.
    batch_size: Dataloader param.
    pin_memory: Dataloader param.
    num_workers: Number of workers used.
    include_spatial_cell_emb: If `True`, also return gene embeddings for spatially contextualized cell embedding.

    Returns
    -------
    output_gene_embed : dict
        Dictionary mapping embedding names to numpy arrays.
    """
    # Load token dictionary
    token_dictionary_file_path = Path(model_folder_path) / 'token_dictionary.pkl'
    with open(token_dictionary_file_path, 'rb') as f:
        token_dict = pickle.load(f)

    neighborhood_gene_ids = [token_dict[ensg] for ensg in neighborhood_gene_ensembl_id]
    cell_gene_ids         = [token_dict[ensg] for ensg in cell_gene_ensembl_id]

    # Create reverse mapping from token id to ensembl id
    id_to_ensembl = {v: k for k, v in token_dict.items()}

    if include_spatial_cell_emb:
        all_cell_gene_emb_dict, all_neighborhood_gene_emb_dict, all_spatial_cell_gene_emb_dict = gene_embed_dataset(
            dataset=dataset,
            model_folder_path=model_folder_path,
            emb_layer=emb_layer,
            cell_gene_ids=list(set(cell_gene_ids)),
            neighborhood_gene_ids=list(set(neighborhood_gene_ids)),
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            return_gene=True,
            compute_cosine_with_list=['neighborhood'],
            include_spatial_cell_emb=include_spatial_cell_emb,
            description='GETTING GENE EMBEDDINGS'
        )
    else:
        all_cell_gene_emb_dict, all_neighborhood_gene_emb_dict = gene_embed_dataset(
            dataset=dataset,
            model_folder_path=model_folder_path,
            emb_layer=emb_layer,
            cell_gene_ids=list(set(cell_gene_ids)),
            neighborhood_gene_ids=list(set(neighborhood_gene_ids)),
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            return_gene=True,
            compute_cosine_with_list=['neighborhood'],
            include_spatial_cell_emb=include_spatial_cell_emb,
            description='GETTING GENE EMBEDDINGS'
        )
    output_gene_embed = {}        
    for gene_id in cell_gene_ids:
        ensg = id_to_ensembl[gene_id]
        output_gene_embed[f"cell_emb_gene{ensg}"] = np.array(torch.cat(
            all_cell_gene_emb_dict[gene_id],
            dim=0).cpu())
        del all_cell_gene_emb_dict[gene_id]
        if include_spatial_cell_emb:
            output_gene_embed[f"spatial_cell_emb_gene{ensg}"] = np.array(torch.cat(
                all_spatial_cell_gene_emb_dict[gene_id],
                dim=0).cpu())
            del all_spatial_cell_gene_emb_dict[gene_id]
    for gene_id in neighborhood_gene_ids:
        ensg = id_to_ensembl[gene_id]
        output_gene_embed[f"neighborhood_emb_gene{ensg}"] = np.array(torch.cat(
            all_neighborhood_gene_emb_dict[gene_id],
            dim=0).cpu())
        del all_neighborhood_gene_emb_dict[gene_id]
    return output_gene_embed


@torch.inference_mode()
def get_average_gene_embed(
    dataset: Dataset,
    model_folder_path: str,
    emb_layer: int | None = None,
    cell_gene_ensembl_id: list = [],
    neighborhood_gene_ensembl_id: list = [],
    batch_size: int = 128,
    pin_memory: bool = False,
    num_workers: int = 12,
    include_spatial_cell_emb: bool = True,
) -> dict[str, np.ndarray]:
    """
    Retrieve average gene embeddings for each gene per dataset.

    Parameters
    -----------
    dataset: Tokenized huggingface dataset.
    model_folder_path: Path to the folder containing the model config, token dictionary, and normalization factors.
    emb_layer: Layer for which to retrieve the embedding.
    cell_gene_ensembl_id: List with gene IDs for which cell gene embeddings will be retrieved.
    neighborhood_gene_ensembl_id: List with gene IDs for which neighborhood gene embeddings will be retrieved.
    batch_size: Dataloader param.
    pin_memory: Dataloader param.
    num_workers: Number of workers used.
    include_spatial_cell_emb: If `True`, also return average gene embeddings for spatially contextualized cell embedding.

    Returns
    -------
    output_average_gene_embed : dict
        Dictionary mapping average embedding/statistics names to numpy arrays.
    """
    # Load token dictionary
    token_dictionary_file_path = Path(model_folder_path) / 'token_dictionary.pkl'
    with open(token_dictionary_file_path, 'rb') as f:
        token_dict = pickle.load(f)

    neighborhood_gene_ids = [token_dict[ensg] for ensg in neighborhood_gene_ensembl_id]
    cell_gene_ids         = [token_dict[ensg] for ensg in cell_gene_ensembl_id]

    if include_spatial_cell_emb:
        all_cell_gene_emb_per_data_dict, all_neighborhood_gene_emb_per_data_dict, all_spatial_cell_gene_emb_per_data_dict = gene_embed_dataset(
            dataset=dataset,
            model_folder_path=model_folder_path,
            emb_layer=emb_layer,
            cell_gene_ids=list(set(cell_gene_ids)),
            neighborhood_gene_ids=list(set(neighborhood_gene_ids)),
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            return_gene_per_data=True,
            compute_cosine_with_list=['neighborhood'],
            include_spatial_cell_emb=include_spatial_cell_emb,
            description='GETTING AVERAGE GENE EMBEDDINGS'
        )
    else:
        all_cell_gene_emb_per_data_dict, all_neighborhood_gene_emb_per_data_dict = gene_embed_dataset(
            dataset=dataset,
            model_folder_path=model_folder_path,
            emb_layer=emb_layer,
            cell_gene_ids=list(set(cell_gene_ids)),
            neighborhood_gene_ids=list(set(neighborhood_gene_ids)),
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            return_gene_per_data=True,
            compute_cosine_with_list=['neighborhood'],
            include_spatial_cell_emb=include_spatial_cell_emb,
            description='GETTING AVERAGE GENE EMBEDDINGS'
        )
    cell_gene_emb_features = []
    cell_gene_emb_counts = []
    if include_spatial_cell_emb:
        spatial_cell_gene_emb_features = []
        spatial_cell_gene_emb_counts = []

    neighborhood_gene_emb_features = []
    neighborhood_gene_emb_counts = []

    for gene_id in cell_gene_ids:
        if gene_id in all_cell_gene_emb_per_data_dict.keys():
            sum_emb = all_cell_gene_emb_per_data_dict[gene_id][0].numpy()
            count_emb = all_cell_gene_emb_per_data_dict[gene_id][1].numpy()
            cell_gene_emb_features.append((sum_emb / count_emb).reshape(1, -1))
            cell_gene_emb_counts.append(count_emb.reshape(1, -1))
        if include_spatial_cell_emb and gene_id in all_spatial_cell_gene_emb_per_data_dict.keys():
            spatial_sum_emb = all_spatial_cell_gene_emb_per_data_dict[gene_id][0].numpy()
            spatial_count_emb = all_spatial_cell_gene_emb_per_data_dict[gene_id][1].numpy()
            spatial_cell_gene_emb_features.append((spatial_sum_emb / spatial_count_emb).reshape(1, -1))
            spatial_cell_gene_emb_counts.append(spatial_count_emb.reshape(1, -1))

    for gene_id in neighborhood_gene_ids:
        if gene_id in all_neighborhood_gene_emb_per_data_dict.keys():
            sum_emb = all_neighborhood_gene_emb_per_data_dict[gene_id][0].numpy()
            count_emb = all_neighborhood_gene_emb_per_data_dict[gene_id][1].numpy()
            neighborhood_gene_emb_features.append((sum_emb / count_emb).reshape(1, -1))
            neighborhood_gene_emb_counts.append(count_emb.reshape(1, -1))

    output_average_gene_embed = {}        

    # Concatenate all features, sums, and counts into single numpy arrays
    if cell_gene_emb_features:
        output_average_gene_embed['cell_gene_emb_average_per_data'] = np.concatenate(cell_gene_emb_features, axis=0)
        output_average_gene_embed['cell_gene_emb_counts_per_data'] = np.concatenate(cell_gene_emb_counts, axis=0)
    else:
        output_average_gene_embed['cell_gene_emb_average_per_data'] =np.array([])
        output_average_gene_embed['cell_gene_emb_counts_per_data'] = np.array([])

    if include_spatial_cell_emb:
        if spatial_cell_gene_emb_features:
            output_average_gene_embed['spatial_cell_gene_emb_average_per_data'] = np.concatenate(spatial_cell_gene_emb_features, axis=0)
            output_average_gene_embed['spatial_cell_gene_emb_counts_per_data'] = np.concatenate(spatial_cell_gene_emb_counts, axis=0)
        else:
            output_average_gene_embed['spatial_cell_gene_emb_average_per_data'] = np.array([])
            output_average_gene_embed['spatial_cell_gene_emb_counts_per_data'] = np.array([])

    if neighborhood_gene_emb_features:
        output_average_gene_embed['neighborhood_gene_emb_average_per_data'] = np.concatenate(neighborhood_gene_emb_features, axis=0)
        output_average_gene_embed['neighborhood_gene_emb_counts_per_data'] = np.concatenate(neighborhood_gene_emb_counts, axis=0)
    else:
        output_average_gene_embed['neighborhood_gene_emb_average_per_data'] = np.array([])
        output_average_gene_embed['neighborhood_gene_emb_counts_per_data'] = np.array([])
    
    return output_average_gene_embed

@torch.inference_mode()
def get_spatial_score(
    dataset: Dataset,
    model_folder_path: str,
    emb_layer: int | None = None,
    cell_gene_ensembl_id: list = [],
    neighborhood_gene_ensembl_id: list = [],
    batch_size: int = 128,
    pin_memory: bool = False,
    num_workers: int = 12,
    compute_cosine_with_list: list[str] = ["cell", "neighborhood"],
) -> dict:
    """
    Compute and return cosine similarity matrix for specified gene IDs.

    Parameters
    -----------
    dataset: Tokenized huggingface dataset.
    model_folder_path: Path to the folder containing the model config, token dictionary, and normalization factors.
    emb_layer: Layer for which to retrieve the embedding.
    cell_gene_ensembl_id: List with gene IDs for which cell gene embeddings will be retrieved.
    neighborhood_gene_ensembl_id: List with gene IDs for which neighborhood gene embeddings will be retrieved.
    batch_size: Dataloader param.
    pin_memory: Dataloader param.
    num_workers: Number of workers used.
    compute_cosine_with_list: A list that defines the items with which we want to compute cosine similarity. It could have value of 'cell' or/and 'neighborhood'.

    Returns
    -------
    cos_sim_dict : dict
        Dictionary containing cosine similarity statistics as numpy arrays.
    """
    # Check for duplicates
    if len(cell_gene_ensembl_id) != len(set(cell_gene_ensembl_id)):
        raise ValueError("The list cell_gene_ensembl_id has duplication.")
    if len(neighborhood_gene_ensembl_id) != len(set(neighborhood_gene_ensembl_id)):
        raise ValueError("The list neighborhood_gene_ensembl_id has duplication.")
    # Load token dictionary
    token_dictionary_file_path = Path(model_folder_path) / 'token_dictionary.pkl'
    with open(token_dictionary_file_path, 'rb') as f:
        token_dict = pickle.load(f)

    neighborhood_gene_ids = [token_dict[ensg] for ensg in neighborhood_gene_ensembl_id]
    cell_gene_ids         = [token_dict[ensg] for ensg in cell_gene_ensembl_id]
    compute_cosine_with_list=["cell", "neighborhood"]

    cos_sim_dict = gene_embed_dataset(
        dataset=dataset,
        model_folder_path=model_folder_path,
        emb_layer=emb_layer,
        cell_gene_ids=cell_gene_ids,
        neighborhood_gene_ids=neighborhood_gene_ids,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        compute_cosine_with_list=compute_cosine_with_list,
        return_cosine_sim=True,
        description='COMPUTE THE COSINE SIMILARITY SCORE BETWEEN GENES IN THE CELL AND GENES IN THE NEIGHBORHOOD.'

    )
    out_put_cosine_sim_score = {} 
    for compute_cosine_with in compute_cosine_with_list:
        out_put_cosine_sim_score[f"sum_cos_sim_{compute_cosine_with}"] = cos_sim_dict[compute_cosine_with][0].numpy()
        out_put_cosine_sim_score[f"pair_count_{compute_cosine_with}"] = cos_sim_dict[compute_cosine_with][1].numpy()
        out_put_cosine_sim_score[f"cell_count_{compute_cosine_with}"] = cos_sim_dict[compute_cosine_with][2].numpy()
        out_put_cosine_sim_score[f"cos_sim_{compute_cosine_with}"] = out_put_cosine_sim_score[f"sum_cos_sim_{compute_cosine_with}"] / out_put_cosine_sim_score[f"pair_count_{compute_cosine_with}"]
    return out_put_cosine_sim_score
@torch.inference_mode()
def get_emd_distance(
    dataset: Dataset,
    model_folder_path: str,
    emb_layer: int | None = None,
    cell_gene_ensembl_id: list = [],
    neighborhood_gene_ensembl_id: list = [],
    batch_size: int = 128,
    pin_memory: bool = False,
    num_workers: int = 12,
) -> np.ndarray:
    """
    Compute and return distance between cosine similarity of cell_neb and cell_cell matrix.

    Parameters
    -----------
    dataset: Tokenized huggingface dataset.
    model_folder_path: Path to the folder containing the model config, token dictionary, and normalization factors.
    emb_layer: Layer for which to retrieve the embedding.
    cell_gene_ensembl_id: List with gene IDs for which cell gene embeddings will be retrieved.
    neighborhood_gene_ensembl_id: List with gene IDs for which neighborhood gene embeddings will be retrieved.
    batch_size: Dataloader param.
    pin_memory: Dataloader param.
    num_workers: Number of workers used.

    Returns
    -------
    emd_array : np.ndarray
        Numpy array of EMD distances.
    """
    # Check for duplicates
    if len(cell_gene_ensembl_id) != len(set(cell_gene_ensembl_id)):
        raise ValueError("The list cell_gene_ensembl_id has duplication.")
    if len(neighborhood_gene_ensembl_id) != len(set(neighborhood_gene_ensembl_id)):
        raise ValueError("The list neighborhood_gene_ensembl_id has duplication.")
    # Load token dictionary
    token_dictionary_file_path = Path(model_folder_path) / 'token_dictionary.pkl'
    with open(token_dictionary_file_path, 'rb') as f:
        token_dict = pickle.load(f)

    neighborhood_gene_ids = [token_dict[ensg] for ensg in neighborhood_gene_ensembl_id]
    cell_gene_ids         = [token_dict[ensg] for ensg in cell_gene_ensembl_id]

    emd_list, emd_matrix_list = gene_embed_dataset(
        dataset=dataset,
        model_folder_path=model_folder_path,
        emb_layer=emb_layer,
        cell_gene_ids=cell_gene_ids,
        neighborhood_gene_ids=neighborhood_gene_ids,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        compute_cosine_with_list=["cell", "neighborhood"],
        return_distance=True,
        description='COMPUTE EMD DISTANCE BETWEEN GENE IN CELL AND NEIGHBORHOOD'
    )
    
    return np.concatenate(emd_list, axis=0), np.concatenate(emd_matrix_list, axis=0)

@torch.inference_mode()
def average_receptor_embedding(
    dataset: Dataset,
    model_folder_path: str,
    emb_layer: int | None = None,
    cell_gene_ensembl_id: list = [],
    neighborhood_gene_ensembl_id: list = [],
    batch_size: int = 128,
    pin_memory: bool = False,
    num_workers: int = 12,
) -> dict:
    """
    Compute the average embedding for each gene in cell_gene_ensembl_id based on its co-occurrence with genes in neighborhood_gene_ensembl_id.

    For each gene i in cell_gene_ensembl_id and each gene j in neighborhood_gene_ensembl_id, 
    accumulate the sum and count of gene i's embedding when both i and j co-occur in cell and neighborhood, 
    respectively. Return the sum, count, and average for each (i, j) pair.

    Returns
    -------
    result : dict
        Dictionary with keys (i, j) for each gene pair, 
        each containing sum, count, and average embedding for gene i when co-occurring with gene j.
    """
    #check for duplicates
    if len(cell_gene_ensembl_id) != len(set(cell_gene_ensembl_id)):
        raise ValueError("The list cell_gene_ensembl_id has duplication.")
    if len(neighborhood_gene_ensembl_id) != len(set(neighborhood_gene_ensembl_id)):
        raise ValueError("The list neighborhood_gene_ensembl_id has duplication.")

    # Load token dictionary
    token_dictionary_file_path = Path(model_folder_path) / 'token_dictionary.pkl'
    with open(token_dictionary_file_path, 'rb') as f:
        token_dict = pickle.load(f)

    neighborhood_gene_ids = [token_dict[ensg] for ensg in neighborhood_gene_ensembl_id]
    cell_gene_ids         = [token_dict[ensg] for ensg in cell_gene_ensembl_id]
    id_to_ensembl = {v: k for k, v in token_dict.items()}

    # Call gene_embed_dataset with a new flag for receptor averaging
    receptor_sum_count = gene_embed_dataset(
        dataset=dataset,
        model_folder_path=model_folder_path,
        emb_layer=emb_layer,
        cell_gene_ids=cell_gene_ids,
        neighborhood_gene_ids=neighborhood_gene_ids,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        return_receptor_average=True,  # This flag should be handled in gene_embed_dataset
        description='GETTING AVERAGE RECEPTOR EMBEDDINGS'
    )

    # receptor_sum_count is expected to be a dict[(cell_gene_id, neighborhood_gene_id, context)] = (sum, count)
    result = {}
    for key, (sum_emb, count_emb) in receptor_sum_count.items():
        if len(key) == 3:  # New format with context ('present' or 'absent')
            cell_id, neigh_id, context = key
            ensg_cell = id_to_ensembl[cell_id]
            ensg_neigh = id_to_ensembl[neigh_id]
            result_key = (ensg_cell, ensg_neigh, context)
        else:  # Old format without context (backward compatibility)
            cell_id, neigh_id = key
            ensg_cell = id_to_ensembl[cell_id]
            ensg_neigh = id_to_ensembl[neigh_id]
            result_key = (ensg_cell, ensg_neigh)
        
        avg_emb = sum_emb.numpy() / count_emb if count_emb > 0 else np.zeros_like(sum_emb.numpy())
        result[result_key] = {
            'sum': sum_emb.numpy(),
            'count': count_emb,
            'average': avg_emb
        }
    return result
