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
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from datasets import Dataset
from functools import partial
from tqdm import tqdm
from pyensembl import EnsemblRelease

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


_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


@torch.no_grad()
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
          feature_norm: bool = False,
          top_k: int | None = None,
          return_gene: bool=True,
          return_cosine_sim: bool=False,
          compute_cosine_with_list:  list[str] = [],
          return_gene_per_data: bool=False,
          return_gene_marker_score: bool=False,
          returen_distance: bool=False,
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
    feature_norm:
        If `True`, apply feature norm in the last embedding layer.
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
    returen_distance:
        If 'True' will compute and return distance between cosine sim of cell_neb 
        and cell_cell matrix.

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

    if args['data']['precomputed_n_nonzero_tokens']:
        with open(args['data']['precomputed_n_nonzero_tokens'], "rb") as f: 
            n_nonzero_tokens = pickle.load(f)
    else:
        n_nonzero_tokens = None
    
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
        sep_gene_tokens_neb=sep_gene_tokens_neb)

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
            per_block_mask_ratio=per_block_mask_ratio)
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
        n_nonzero_tokens_list=n_nonzero_tokens,
        include_cell_id=True,
        sep_gene_tokens_neb=sep_gene_tokens_neb)

    loader = init_dataloader_and_sampler(
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

            cell_emb_list = []
            neighborhood_emb_list = []
            for emb_layer in emb_layers:
                cell_emb_list.append(return_layer_emb_fn(
                    layer=emb_layer,
                    udata=udata,
                    masks_attention=masks_attention,
                    pad_neighborhood=True).cpu())
                neighborhood_emb_list.append(return_layer_emb_fn(
                    layer=emb_layer,
                    udata=udata,
                    masks_attention=masks_attention,
                    pad_neighborhood=False).cpu())
        
            if feature_norm and (emb_layers[-1] == enc_depth):
                # Normalize last layer like in training # TO DO should this consider inference padding?
                cell_emb_list[-1] = F.layer_norm(cell_emb_list[-1],
                                                 (cell_emb_list[-1].size(-1),))
                neighborhood_emb_list[-1] = F.layer_norm(neighborhood_emb_list[-1],
                                                         (neighborhood_emb_list[-1].size(-1),))

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
                neighborhood_emb = compute_mean_unmasked_emb(n_emb, neighborhood_mask)
            elif agg_type == "weighted_avg":
                cell_weights = compute_unmasked_rank_based_weights(
                    tokens, cell_mask)
                cell_emb = compute_mean_unmasked_emb(
                    c_emb * cell_weights.unsqueeze(-1),
                    cell_mask)
                neighborhood_weights = compute_unmasked_rank_based_weights(
                    tokens, neighborhood_mask)
                neighborhood_emb = compute_mean_unmasked_emb(
                    n_emb * neighborhood_weights.unsqueeze(-1),
                    neighborhood_mask)

            # Concat layer-specific embeddings across batches
            if itr == 0:
                all_cell_emb_list.append([cell_emb])
                all_neighborhood_emb_list.append([neighborhood_emb])
            else:
                all_cell_emb_list[i].append(cell_emb) 
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
                    if len(neighborhood_gene_ids) != 0 and (return_cosine_sim or return_gene_marker_score or returen_distance):
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
                if returen_distance:
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
    if returen_distance:
       adata.obsm['emd_dist'] = np.concatenate(emd_list, axis=0)

    return adata


def harmonize_adata(adata: ad.AnnData,
                    ensembl_release: int=110, # 111
                    ) -> ad.AnnData:
    """
    Harmonize an AnnData object prior to tokenization.

    Parameters
    -----------
    adata:
        An unharmonized AnnData object.

    Returns:
    -----------
    adata:
        A harmonized AnnData object.
    """
    print('==================================================')
    print('STEP 1: ADDING ENSEMBL IDS...')
    print('==================================================')
    print(f'Adding ensembl IDs from release {ensembl_release}...')
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

    adata_gene_names = [gene_name for gene_name in adata.var_names.tolist()]
    adata.var.index = adata_gene_names
    
    harmonized_gene_names = []
    matching_ensembl_ids = []
    for gene_name in adata_gene_names:
        if gene_name in gene_ensembl_map_dict.keys():
            harmonized_gene_names.append(gene_name)
            matching_ensembl_ids.append(gene_ensembl_map_dict[gene_name])
    print(f'Number of genes with matching ensembl IDs: \
          {len(harmonized_gene_names)}.')
    print(f'Number of genes skipped: \
          {len(adata_gene_names) - len(harmonized_gene_names)}.')

    adata = adata[:, adata.var.index.isin(harmonized_gene_names)].copy()
    adata.var = pd.DataFrame(
        index=pd.Index(harmonized_gene_names, name="gene_name"),
        data={"ensembl_id": matching_ensembl_ids})

    # Add dummy values as special values
    print('==================================================')
    print('STEP 2: ADDING SPECIAL VALUES...')
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
                   nproc: int = 4,
                   processing_mode: Literal['sequential',
                                            'parallel'] = 'parallel'
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
    n_proc:
        Number of processes used.
    processing_mode:
        Mode of processing.

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
        token_dictionary_file_path=token_dictionary_file_path)
    dataset_dict = tk._tokenize_adata(adata=adata)
    dataset = tk._create_dataset(
        dataset_dict=dataset_dict,
        use_generator=False,
        cache_directory_path=None,
        keep_in_memory=False)

    columns = list(dataset.features.keys())
    columns.remove("cell_id")
    dataset.set_format(
        type="torch",
        columns=columns,
        output_all_columns=True)
    
    return dataset


@torch.no_grad()
def embed_dataset(dataset: Dataset,
                  model_folder_path: str,
                  emb_layer: int | None = None,
                  agg_excluded_tokens: list[int] | None = None,
                  top_k: int | None = None,
                  batch_size: int = 128,
                  pin_memory: bool = False,
                  num_workers: int = 12,
                  include_spatial_cell_emb: bool = True,
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
        per_block_mask_ratio=model_config['mask']['per_block_mask_ratio'])
        
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
        n_nonzero_tokens_list=None,
        include_cell_id=True,
        sep_gene_tokens_neb=model_config['data']['sep_gene_tokens_neb'])

    # Initialize dataloader
    loader = init_dataloader_and_sampler(
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

            c_emb = return_layer_emb_fn(
                layer=emb_layer,
                udata=udata,
                masks_attention=masks_attention,
                pad_neighborhood=True).cpu()
            n_emb = return_layer_emb_fn(
                layer=emb_layer,
                udata=udata,
                masks_attention=masks_attention,
                pad_neighborhood=False).cpu()
        
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

    return output_embed


def perturb_dataset(dataset: Dataset,
                    perturb_df: pd.DataFrame,
                    model_folder_path: str,
                    seq_len_cell: int = 256,
                    nproc: int = 4,
                    keep_in_memory: bool = False,) -> Dataset:

    def _perturb_example(example,
                        perturb_df: pd.DataFrame,
                        seq_len_cell: int = 256) -> dict:
        example_cell_ids = list(dict.fromkeys(example['cell_ids']))
        
        if not any(cell_id in perturb_df['perturbed_cell_id'].values.tolist() for cell_id in example_cell_ids):
            # No perturbation applied
            return example
            
        for idx, row in perturb_df.iterrows():
            
            perturbed_cell_id = row['perturbed_cell_id']
            
            if row['perturbation_target'] == 'cell':
                if not perturbed_cell_id == example_cell_ids[0]:
                    continue
                else:
                    #print(f"Perturb index cell with ID {example['cell_id']}:")
                    if row['perturbed_gene_token'] == 'all':
                        perturbed_token_idx = torch.arange(0, seq_len_cell)
                    else:
                        perturbed_token_idx = (example['gene_tokens'][:seq_len_cell] == row['perturbed_gene_token']).nonzero(as_tuple=True)[0]
                    if row['perturbation_type'] == 'knockout':
                        example['gene_tokens'][perturbed_token_idx] = 0
                        example['gene_expr'][perturbed_token_idx] = 0.0
                    elif row['perturbation_type'] == 'foldchange':
                        example['gene_expr'][perturbed_token_idx] = example['gene_expr'][perturbed_token_idx] * row['foldchange']                    
                    else:
                        raise ValueError(f'Invalid perturbation type {row["perturbation_type"]}.')
                                            
            elif row['perturbation_target'] == 'neighborhood':
                if not perturbed_cell_id in example_cell_ids[1:]:
                    continue
                else:
                    #print(f"Perturb neighborhood of index cell with ID {example['cell_id']}:")
                    if row['perturbed_gene_token'] == 'all':
                        perturbed_token_idx = torch.arange(seq_len_cell, len(example['gene_tokens']))
                    else:
                        perturbed_token_idx = (example['gene_tokens'][seq_len_cell:] == row['perturbed_gene_token']).nonzero(as_tuple=True)[0]
                    if row['perturbation_type'] == 'knockout':
                        example['gene_tokens'][perturbed_token_idx] = 0
                        example['gene_expr'][perturbed_token_idx] = 0.0
                    elif row['perturbation_type'] == 'foldchange':
                        example['gene_expr'][perturbed_token_idx] = example['gene_expr'][perturbed_token_idx] * row['foldchange']                    
                    else:
                        raise ValueError(f'Invalid perturbation type {row["perturbation_type"]}.')           
            else:
                raise ValueError(f'Invalid perturbation target {row["perturbation_target"]}.')
        
        # Perturbation applied
        return example


    # Load token dictionary
    token_dictionary_file_path = Path(model_folder_path) / 'token_dictionary.pkl'
    with open(token_dictionary_file_path, 'rb') as f:
        token_dict = pickle.load(f)
    
    perturb_df['perturbed_gene_token'] = perturb_df['perturbed_ensembl_id'].apply(lambda x: x if x == 'all' else token_dict[x])

    perturb_func = partial(
        _perturb_example,
        perturb_df=perturb_df,
        seq_len_cell=seq_len_cell)
    
    # Apply perturbation example-wise
    perturbed_dataset = dataset.map(
        perturb_func,
        num_proc=nproc,
        keep_in_memory=keep_in_memory)    
    
    return perturbed_dataset


@torch.no_grad()
def harmonize_tokenize_embed_pipeline(
        adata: ad.AnnData,
        model_folder_path: str,
        gene_perturb_df: pd.DataFrame | None = None,               
        nproc: int = 4,
        processing_mode: Literal['sequential',
                                 'parallel'] = 'parallel',
        save_dataset_path: Path | str | None = None,
        num_shards: int = 32,
        emb_layer: int | None = None,
        cell_gene_ids: list = [],
        neighborhood_gene_ids: list = [],
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
    cell_gene_ids:
        List with gene IDs for which cell gene embeddings will be retrieved.
    neighborhood_gene_ids:
        List with gene IDs for which neighborhood gene embeddings will be
        retrived.
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
    print('====================================================================================================')
    print('                                         HARMONIZE DATA                                             ')
    print('====================================================================================================')    
    adata = harmonize_adata(adata)

    print('====================================================================================================')
    print('                                          TOKENIZE DATA                                             ')
    print('====================================================================================================')  
    dataset = tokenize_adata(
        adata=adata,
        model_folder_path=model_folder_path,             
        nproc=nproc,
        processing_mode=processing_mode)

    if save_dataset_path:
        dataset.save_to_disk(
            save_dataset_path,
            num_shards=num_shards)

    print('====================================================================================================')
    print('                                            EMBED DATA                                              ')
    print('====================================================================================================')  
    output_embed = embed_dataset(
        dataset=dataset,
        model_folder_path=model_folder_path,
        emb_layer=emb_layer,
        cell_gene_ids=cell_gene_ids,
        neighborhood_gene_ids=neighborhood_gene_ids,
        agg_excluded_tokens=agg_excluded_tokens,
        top_k=top_k,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers)

    # Add embeddings to adata
    for key, values in output_embed.items():
        adata.obsm[key] = values

    return adata


@torch.no_grad()
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
                  returen_distance: bool=False,
                  return_cosine_sim: bool=False,
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
    returen_distance:
        If 'True' will compute and return distance between cosine sim of cell_neb 
        and cell_cell matrix.
    return_cosine_sim: 
        If 'True' will compute and return cosine_sim matrix.
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
        per_block_mask_ratio=model_config['mask']['per_block_mask_ratio'])
        
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
        n_nonzero_tokens_list=None,
        include_cell_id=True,
        sep_gene_tokens_neb=model_config['data']['sep_gene_tokens_neb'])

    # Initialize dataloader
    loader = init_dataloader_and_sampler(
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
    cos_sim_dict = {}
    emd_list = []
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

            c_emb = return_layer_emb_fn(
                layer=emb_layer,
                udata=udata,
                masks_attention=masks_attention,
                pad_neighborhood=True).cpu()
            n_emb = return_layer_emb_fn(
                layer=emb_layer,
                udata=udata,
                masks_attention=masks_attention,
                pad_neighborhood=False).cpu()
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
                if len(neighborhood_gene_ids) != 0 and (return_cosine_sim  or returen_distance):
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
            if returen_distance:
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
    # last layer
    if return_gene:
        return all_cell_gene_emb_dict, all_neighborhood_gene_emb_dict
    if return_gene_per_data:
        return all_cell_gene_emb_per_data_dict, all_neighborhood_gene_emb_per_data_dict
    if return_cosine_sim:
        return cos_sim_dict
    if returen_distance:
        return emd_list

@torch.no_grad()
def get_gene_embed(
    dataset: Dataset,
    model_folder_path: str,
    emb_layer: int | None = None,
    cell_gene_ensembl_id: list = [],
    neighborhood_gene_ensembl_id: list = [],
    batch_size: int = 128,
    pin_memory: bool = False,
    num_workers: int = 12,
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
        description='GETTING GENE EMBEDDINGS'
    )
    output_gene_embed = {}        
    for gene_id in cell_gene_ids:
        ensg = id_to_ensembl[gene_id]
        output_gene_embed[f"cell_emb_gene{ensg}"] = np.array(torch.cat(
            all_cell_gene_emb_dict[gene_id],
            dim=0).cpu())
        del all_cell_gene_emb_dict[gene_id]
    for gene_id in neighborhood_gene_ids:
        ensg = id_to_ensembl[gene_id]
        output_gene_embed[f"neighborhood_emb_gene{ensg}"] = np.array(torch.cat(
            all_neighborhood_gene_emb_dict[gene_id],
            dim=0).cpu())
        del all_neighborhood_gene_emb_dict[gene_id]
    return output_gene_embed


@torch.no_grad()
def get_average_gene_embed(
    dataset: Dataset,
    model_folder_path: str,
    emb_layer: int | None = None,
    cell_gene_ensembl_id: list = [],
    neighborhood_gene_ensembl_id: list = [],
    batch_size: int = 128,
    pin_memory: bool = False,
    num_workers: int = 12,
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

    all_cell_gene_emb_per_data_dict, all_neighborhood_gene_emb_per_data_dict= gene_embed_dataset(
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
        description='GETTING AVERAGE GENE EMBEDDINGS'
    )
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

    output_average_gene_embed = {}        

        # Concatenate all features, sums, and counts into single numpy arrays
    output_average_gene_embed['cell_gene_emb_average_per_data'] = np.concatenate(cell_gene_emb_features, axis=0)
    output_average_gene_embed['cell_gene_emb_counts_per_data'] = np.concatenate(cell_gene_emb_counts, axis=0)

    output_average_gene_embed['neighborhood_gene_emb_average_per_data'] = np.concatenate(neighborhood_gene_emb_features, axis=0)
    output_average_gene_embed['neighborhood_gene_emb_counts_per_data'] = np.concatenate(neighborhood_gene_emb_counts, axis=0)
    return output_average_gene_embed

@torch.no_grad()
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
@torch.no_grad()
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

    emd_list = gene_embed_dataset(
        dataset=dataset,
        model_folder_path=model_folder_path,
        emb_layer=emb_layer,
        cell_gene_ids=cell_gene_ids,
        neighborhood_gene_ids=neighborhood_gene_ids,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        compute_cosine_with_list=["cell", "neighborhood"],
        returen_distance=True,
        description='COMPUTE EMD DISTANCE BETWEEN GENE IN CELL AND NEIGHBORHOOD'
    )
    
    return np.concatenate(emd_list, axis=0)
