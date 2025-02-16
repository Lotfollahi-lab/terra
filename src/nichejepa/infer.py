import copy
import logging
import os
import sys
import yaml
from collections import defaultdict
from typing import List, Literal, Optional

import anndata as ad
import numpy as np
import pandas as pd
import pickle
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from .datasets.cell_datasets import CellBaseDataset, make_cell_dataset
from .datasets.dataloaders import init_dataloader_and_sampler
from .helper import init_model, load_checkpoint
from .masks.block_masking  import BlockMaskCollator
from .masks.random_masking import RandomMaskCollator
from .utils.distributed import init_distributed
from .utils.embedding import (create_binary_selection_mask,
                              compute_mean_unmasked_emb,
                              compute_unmasked_rank_based_weights,
                              collect_adata_from_folder,
                              retrieve_gene_emb,
                              compute_cosine_similarity_components)
from .utils.logging import CSVLogger


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
          cell_gene_ids: List=[],
          neighborhood_gene_ids: List=[],
          agg_type: Literal['cls_cell',
                            'cls_neighborhood',
                            'avg',
                            'weighted_avg']='avg',
          masked_tokens: Optional[List[int]]=None,
          agg_excluded_tokens: Optional[List[int]]=None,
          feature_norm: bool=False,
          check_point: int=None,
          top_k: int=None,
          return_all: bool=True,
          ) -> ad.AnnData:
    """
    Use a trained model for inference. Run forward pass on a given dataset and
    return cell, neighborhood and (optionally) gene embeddings (cell and
    neighborhood gene embeddings).

    Parameters
    -----------
    args:
        Dictionary containing the hyperparameters from the config file.
    dataset:
        Cell dataset for which embeddings will be inferred.
    cell_gene_ids:
        List with gene IDs for which cell gene embeddings will be retrived.
    neighborhood_gene_ids:
        List with gene IDs for which neighborhood gene embeddings will be
        retrived.
    agg_type:
        Specifies how (aggregated) cell and neighborhood embeddings are computed
        from individual gene embeddings.
    masked_tokens:
        List of tokens to be masked by the attention mask during inference.
    agg_excluded_tokens:
        List of tokens to be excluded from the aggregation.
    feature_norm:
        If 'True', apply feature norm in the last embedding layer.
    check_point:
        If load from check_point or not.
    top_k:
        Include top_k genes or not.
    return_all: 
        If return all inferenced information or not.
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
    gt_type = args['meta']['gt_type']
    enc_depth = args['meta']['enc_depth']
    enc_emb_dim = args['meta']['enc_emb_dim']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    special_tokens = args['meta']['special_tokens']
    pos_learnable = args['meta']['pos_learnable']
    seg_learnable = args['meta']['seg_learnable']
    use_bfloat16 = args['meta']['use_bfloat16']
    use_flash_attention = args['meta']['use_flash_attention']

    dataset_name = args['data']['dataset_name']
    raw_data_folder_path = args['data']['raw_data_folder_path']
    token_dict_folder_path = args['data']['token_dict_folder_path']
    batch_size = args['data']['batch_size']
    pin_memory = args['data']['pin_memory']
    num_workers = args['data']['num_workers']
    tokenizer_type = args['data']['tokenizer_type']
    seq_len_cell = args['data']['seq_len_cell']
    seq_len_neighborhood = args['data']['seq_len_neighborhood']
    n_segments = args['data']['n_segments']

    n_contexts = args['mask']['n_contexts']
    n_targets = args['mask']['n_targets']
    block_masking = args['mask']['block_masking']
    context_mask_size = args['mask']['context_mask_size']
    target_mask_size = args['mask']['target_mask_size']
    per_block_mask_ratio = args['mask']['per_block_mask_ratio']
    if args['mask']['controlled_attention_pattern'] is not None:
        controlled_attention_pattern = torch.tensor(args['mask']['controlled_attention_pattern'])
    else:
        controlled_attention_pattern = args['mask']['controlled_attention_pattern']
    restrict_special_attention = args['mask']['restrict_special_attention']

    if args['data']['precomputed_n_nonzero_tokens']:
        with open(args['data']['precomputed_n_nonzero_tokens'], "rb") as f: 
            n_nonzero_tokens= pickle.load(f)
    else:
        n_nonzero_tokens = None
        print(n_nonzero_tokens)

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
        special_tokens = ['cls_cell', 'cls_neighborhood'] + special_tokens
    elif tokenizer_type == 'cell_graph':
        special_tokens = [
                f'cls_{i}' for i in range(n_segments)] + special_tokens

    max_cls_tokens = sum('cls' in token for token in special_tokens)
    # Get token sequence length and number of special tokens
    n_special_tokens = len(special_tokens)
    seq_len = seq_len_cell + seq_len_neighborhood + n_special_tokens

    # Set the folder for saving extracted features
    save_folder = f"{load_folder_path}/extracted_features"
    feature_path = f"{save_folder}/"

    os.makedirs(save_folder, exist_ok=True)
    dump = os.path.join(save_folder, f'params.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)

    # Initialize torch distributed backend
    world_size, rank = init_distributed()

    # Define checkpointing path
    if check_point is None:
       latest_path = os.path.join(load_folder_path, f'{tag}-latest.pth.tar')
    else:
       latest_path = os.path.join(load_folder_path, f'{tag}-ep{check_point}.pth.tar')
    load_path = (os.path.join(load_folder_path, r_file) if r_file is not None
        else latest_path)

    # Initialize encoder, predictor, and target encoder
    encoder, predictor = init_model(
        gt_type=gt_type,
        device=device,
        vocab_size=vocab_size,
        seq_len=seq_len,
        max_cls_tokens=max_cls_tokens,
        max_special_tokens=max_special_tokens,
        n_special_tokens=n_special_tokens,
        n_segments=n_segments,
        n_special_values=n_special_values,
        enc_emb_dim=enc_emb_dim,
        enc_depth=enc_depth,
        pred_emb_dim=pred_emb_dim,
        pred_depth=pred_depth,
        pos_learnable=pos_learnable,
        seg_learnable=seg_learnable,
        use_flash_attention=use_flash_attention)
    target_encoder = copy.deepcopy(encoder)

    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)

    # Initialize mask collator
    if block_masking:
       mask_collator = BlockMaskCollator(
            n_targets=n_targets,
            n_contexts=n_contexts,
            seq_len_cell=seq_len_cell,
            seq_len_neighborhood=seq_len_neighborhood,
            max_special_tokens=max_special_tokens,
            n_special_tokens=n_special_tokens,
            max_cls_tokens=max_cls_tokens,
            per_block_mask_ratio=per_block_mask_ratio,
            controlled_attention_pattern=controlled_attention_pattern,
            restrict_special_attention=restrict_special_attention)
    else:
        mask_collator = RandomMaskCollator(
            n_targets=n_targets,
            n_contexts=n_contexts,
            seq_len_cell=seq_len_cell,
            seq_len_neighborhood=seq_len_neighborhood,
            n_special_tokens=n_special_tokens,
            target_mask_size=target_mask_size,
            context_mask_size=context_mask_size,)

    # Initialize train and test datasets, dataloaders and samplers
    cell_dataset = make_cell_dataset(
        dataset=dataset,
        vocab_size=vocab_size,
        seq_len_cell=seq_len_cell,
        seq_len_neighborhood=seq_len_neighborhood,
        max_cls_tokens=max_cls_tokens,
        max_special_tokens=max_special_tokens,
        tokenizer_type=tokenizer_type,
        gt_type=gt_type,
        special_tokens=special_tokens,
        sampling_strategy=None,
        n_nonzero_tokens_list=n_nonzero_tokens)


    loader = init_dataloader_and_sampler(
        cell_dataset=cell_dataset,
        batch_size=batch_size,
        distributed=False,
        world_size=world_size,
        rank=rank,
        collate_fn=mask_collator,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=False)
    
    _, _, target_encoder, _, _, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=None,
            scaler=None)
   
    # Retrieve embeddings
    target_encoder.eval()

    all_cell_ids = []
    all_cell_emb_list = []
    all_neighborhood_emb_list = []
    all_cell_gene_emb_dict = {}
    all_neighborhood_gene_emb_dict = {}
    sum_cos_sim, count  = None, None
    for itr, (udata, _, _, masks_attention, _, _) in tqdm(enumerate(loader)):
        # Load gene tokens and segmentation label to the specified device
        tokens = udata[0].to(device, non_blocking=True)
        segments = udata[1].to(device, non_blocking=True)
        positions = udata[2].to(device, non_blocking=True)
        counts = udata[3].to(device, non_blocking=True)
        masks_attention = masks_attention.to(device, non_blocking=True)

        # Collect cell IDs to join metadata
        all_cell_ids.extend(udata[-1])

        # Retrieve gene embeddings from different layers
        with torch.cuda.amp.autocast(dtype=torch.bfloat16,
                                     enabled=args['meta']['use_bfloat16']):

            if masked_tokens is not None:
                mask_indices = torch.isin(
                    tokens,
                    torch.tensor(masked_tokens, device=tokens.device)
                    ).unsqueeze(1).unsqueeze(1).expand(
                        -1,-1, tokens.shape[-1], -1)
                masks_attention[mask_indices] = 0

            if gt_type == 'rank':
                emb_list = target_encoder.module.return_multi_layer_emb(
                    positions=positions,
                    segments=segments,
                    tokens=tokens,
                    masks_attention=masks_attention)
            elif gt_type == 'counts':
                emb_list = target_encoder.module.return_multi_layer_emb(
                    tokens=tokens,
                    segments=segments,
                    counts=counts,
                    masks_attention=masks_attention)
        
            if feature_norm:
                # Normalize last layer like in training
                emb_list[-1] = F.layer_norm(emb_list[-1],
                                            (emb_list[-1].size(-1),))

        # Aggregate gene embeddings into cell and neighborhood embeddings
        for i, emb in enumerate(emb_list):
            # Keep only <cls> token; at the moment there is only 1 <cls> token
            if agg_type == 'cls':
                cell_mask = create_binary_selection_mask(
                    tokens,
                    selection_type='cls_0',
                    seq_len_cell=seq_len_cell,
                    n_special_tokens=n_special_tokens,
                    max_cls_tokens=max_cls_tokens,
                    top_k = top_k,
                    n_segments = n_segments)
                if tokenizer_type == 'cell_neighborhood':
                    neighborhood_mask = create_binary_selection_mask(
                        tokens,
                        selection_type='cls_1',
                        seq_len_cell=seq_len_cell,
                        n_special_tokens=n_special_tokens,
                        max_cls_tokens=max_cls_tokens,
                        top_k = top_k,
                        n_segments = n_segments)

                elif tokenizer_type == 'cell_graph':
                    neighborhood_mask = create_binary_selection_mask(
                        tokens,
                        selection_type='cls_all',
                        seq_len_cell=seq_len_cell,
                        n_special_tokens=n_special_tokens,
                        max_cls_tokens=max_cls_tokens,
                        top_k = top_k,
                        n_segments = n_segments)

                cell_emb = compute_mean_unmasked_emb(emb,
                                                     cell_mask)
                neighborhood_emb = compute_mean_unmasked_emb(
                    emb,
                    neighborhood_mask)
            # Keep elements relevant to cell embedding
            elif (agg_type == "avg") or (agg_type == "weighted_avg"):
                cell_mask = create_binary_selection_mask(
                    tokens,
                    selection_type="agg_cell",
                    excluded_tokens=agg_excluded_tokens,
                    seq_len_cell=seq_len_cell,
                    n_special_tokens=n_special_tokens,
                    max_cls_tokens=max_cls_tokens,
                    top_k = top_k,
                    n_segments = n_segments)

                if tokenizer_type == 'cell_neighborhood':
                    neighborhood_mask = create_binary_selection_mask(
                        tokens,
                        selection_type="agg_neighborhood",
                        excluded_tokens=agg_excluded_tokens,
                        seq_len_cell=seq_len_cell,
                        n_special_tokens=n_special_tokens,
                        max_cls_tokens=max_cls_tokens,
                        top_k = top_k,
                        n_segments = n_segments)
                elif tokenizer_type == 'cell_graph':
                    neighborhood_mask = create_binary_selection_mask(
                        tokens,
                        selection_type="agg_graph",
                        excluded_tokens=agg_excluded_tokens,
                        seq_len_cell=seq_len_cell,
                        n_special_tokens=n_special_tokens,
                        max_cls_tokens=max_cls_tokens,
                        top_k = top_k,
                        n_segments = n_segments)                

                if agg_type == 'avg':
                    cell_emb = compute_mean_unmasked_emb(emb,
                                                         cell_mask)
                    neighborhood_emb = compute_mean_unmasked_emb(
                        emb,
                        neighborhood_mask)
                elif agg_type == "weighted_avg":
                    cell_weights = compute_unmasked_rank_based_weights(
                        tokens, cell_mask)
                    cell_emb = compute_mean_unmasked_emb(
                        emb * cell_weights.unsqueeze(-1),
                        cell_mask)
                    neighborhood_weights = compute_unmasked_rank_based_weights(
                        tokens, neighborhood_mask)
                    neighborhood_emb = compute_mean_unmasked_emb(
                        emb * neighborhood_weights.unsqueeze(-1),
                        neighborhood_mask)

            # Concat layer-specific embeddings across batches
            if itr == 0:
                all_cell_emb_list.append([cell_emb])
                all_neighborhood_emb_list.append([neighborhood_emb])
            else:
                all_cell_emb_list[i].append(cell_emb) 
                all_neighborhood_emb_list[i].append(neighborhood_emb)

            # Store cell and neighborhood gene embeddings of last layer
            if i == (len(emb_list) - 1):
                if len(cell_gene_ids)!=0 and itr ==0:
                    cell_embs = torch.zeros((emb.shape[0], len(cell_gene_ids), emb.shape[-1]))
                    neb_embs = torch.zeros((emb.shape[0], len(neighborhood_gene_ids), emb.shape[-1]))
                    cell_presence = torch.zeros((emb.shape[0], len(cell_gene_ids)))
                    neb_presence = torch.zeros((emb.shape[0], len(neighborhood_gene_ids)))
                elif len(cell_gene_ids)!=0:
                    cell_embs.zero_()
                    neb_embs.zero_()
                    cell_presence.zero_()
                    neb_presence.zero_()
                for j, gene_id in enumerate(cell_gene_ids):
                    gene_presence, gene_indices = retrieve_gene_emb(
                        tokens=tokens,
                        gene_id=gene_id,
                        gene_type="cell",
                        seq_len_cell=seq_len_cell,
                        n_special_tokens=n_special_tokens)
                    rows = torch.arange(emb.shape[0])
                     # Only update for sequences where presence is True.
                    gene_presence = gene_presence.cpu()
                    gene_indices = gene_indices.cpu()
                    cell_embs[rows[gene_presence], j, :] = emb.cpu()[rows[gene_presence], gene_indices[gene_presence], :]
                    cell_presence[:, j] = gene_presence.float()
                    if itr == 0 and return_all:
                        all_cell_gene_emb_dict[gene_id] = [cell_embs[:, j, :]]
                    elif return_all:
                        all_cell_gene_emb_dict[gene_id].append(cell_embs[:, j, :])
                for j, gene_id in enumerate(neighborhood_gene_ids):
                    gene_emb, gene_presence = retrieve_gene_emb(
                        tokens=tokens,
                        emb=emb,
                        gene_id=gene_id,
                        gene_type="neighborhood",
                        seq_len_cell=seq_len_cell,
                        n_special_tokens=n_special_tokens,
                        aggregate_multiple=True)
                    neb_embs[:, j, :] = gene_emb.cpu()
                    neb_presence[:, j] = gene_presence.float()
                    if itr == 0 and return_all:
                        all_neighborhood_gene_emb_dict[gene_id] = [neb_embs[:, j, :]]
                    elif return_all:
                        all_neighborhood_gene_emb_dict[gene_id].append(neb_embs[:, j, :])                  
                if itr == 0 and len(cell_gene_ids)!=0:
                    sum_cos_sim, count =compute_cosine_similarity_components(cell_embs, neb_embs, cell_presence, neb_presence)
                elif len(cell_gene_ids)!=0:
                    sum_cos_sim_temp, count_temp =compute_cosine_similarity_components(cell_embs, neb_embs, cell_presence, neb_presence)
                    sum_cos_sim += sum_cos_sim_temp
                    count += count_temp
        np.save("sim_new.npy", sum_cos_sim.cpu())
        np.save("count_new.npy", count.cpu())

    if return_all is False:
        return sum_cos_sim, count
    adata = ad.AnnData(
        obs=pd.DataFrame({'cell_id': all_cell_ids},
        index=range(len(all_cell_ids))))    
    # Add metadata
    adata_metadata = collect_adata_from_folder(raw_data_folder_path)
    adata_metadata_subset = adata_metadata[
        adata_metadata.obs['cell_id'].isin(adata.obs['cell_id'])]
    merged_obs = pd.merge(adata.obs,
                          adata_metadata_subset.obs,
                          on='cell_id')
    adata.obs = merged_obs.set_index('cell_id')
    adata_metadata_subset.obs = adata_metadata_subset.obs.set_index('cell_id')
    adata_metadata_subset = adata_metadata_subset[adata.obs.index, :].copy()
    adata.obsm['spatial'] = adata_metadata_subset.obsm['spatial']
   
    # Store cell and neighborhood embeddings of all observations across layers  
    for i in range(len(all_cell_emb_list)):
        adata.obsm[f"cell_emb_layer_{i}"] = np.array(torch.cat(
            all_cell_emb_list[i],
            dim=0).cpu())
        adata.obsm[f"neighborhood_emb_layer_{i}"] = np.array(torch.cat(
            all_neighborhood_emb_list[i],
            dim=0).cpu())

    # Store cell and neighborhood gene embeddings of all observations in the
    # last layer
    for gene_id in cell_gene_ids:
        adata.obsm[f"cell_emb_gene{gene_id}"] = np.array(torch.cat(
            all_cell_gene_emb_dict[gene_id],
            dim=0).cpu())
    for gene_id in neighborhood_gene_ids:
        adata.obsm[f"neighborhood_emb_gene{gene_id}"] = np.array(torch.cat(
            all_neighborhood_gene_emb_dict[gene_id],
            dim=0).cpu())

    return adata, sum_cos_sim, count
