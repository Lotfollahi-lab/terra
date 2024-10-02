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
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from .datasets.cell_datasets import CellBaseDataset, make_cell_dataset
from .datasets.dataloaders import init_dataloader_and_sampler
from .helper import init_model, load_checkpoint
from .masks.multigene import MaskCollator
from .masks.segment_masking  import SegmentMaskCollator
from .utils.distributed import init_distributed
from .utils.embedding import (create_binary_selection_mask,
                              compute_mean_unmasked_emb,
                              compute_unmasked_rank_based_weights,
                              collect_adata_from_folder,
                              retrieve_gene_emb)
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
          agg_excluded_tokens: Optional[List]=None,
          feature_norm: bool=False
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
    agg_excluded_tokens:
        List of tokens to be excluded from the aggregation.
    feature_norm:
        If 'True', apply feature norm in the last embedding layer.

    Returns
    -----------
    adata:
        An AnnData object with the stored embeddings and labels.
    """
    # ----------------------------- #
    #  Load params from config file
    # ----------------------------- #
    # Load meta params
    use_bfloat16 = args['meta']['use_bfloat16']
    r_file = args['meta']['read_checkpoint']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    enc_depth = args['meta']['enc_depth']
    enc_emb_dim = args['meta']['enc_emb_dim']
    special_tokens = args['meta']['special_tokens']
    pos_learnable = args['meta']['pos_learnable']
    seg_learnable = args['meta']['seg_learnable']

    # Load data params
    raw_data_folder_path = args['data']['raw_data_folder_path']
    batch_size = args['data']['batch_size']
    vocab_size = args['data']['vocab_size']
    pin_memory = args['data']['pin_memory']
    num_workers = args['data']['num_workers']
    dataset_name = args['data']['dataset_name']
    tokenizer_type = args['data']['tokenizer_type']
    seq_len_cell = args['data']['seq_len_cell']
    seq_len_neighborhood = args['data']['seq_len_neighborhood']
    n_segments = args['data']['n_segments']

    # Load optimization params
    num_epochs = args['optimization']['epochs']

    # Load mask params
    n_targets = args['mask']['n_targets']
    n_contexts = args['mask']['n_contexts']
    target_mask_size = args['mask']['target_mask_size']
    context_mask_size = args['mask']['context_mask_size']
    segment_masking = args['mask']['segment_masking']
    per_segment_mask_ratio = args['mask']['per_segment_mask_ratio']
    # ----------------------------- #

    # ------------------------------------ #
    #  Load model and initialize data loader
    # ------------------------------------ #
    # Set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # Initialize torch distributed backend
    world_size, rank = init_distributed()
    
    # Get token sequence length and number of special tokens
    n_special_tokens = len(special_tokens)
    seq_len = seq_len_cell + seq_len_neighborhood + n_special_tokens

    # Set the folder for saving extracted features
    save_folder = f"{load_folder_path}/extracted_features"
    feature_path = f"{save_folder}/"

    os.makedirs(save_folder, exist_ok=True)
    tag = args['logging']['write_tag']
    dump = os.path.join(save_folder, f'params.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)

    # Define checkpointing path
    latest_path = os.path.join(load_folder_path, f'{tag}-latest.pth.tar')
    load_path = (os.path.join(load_folder_path, r_file) if r_file is not None 
        else latest_path)

    # Initialize encoder, predictor, and target encoder
    encoder, predictor = init_model(
        device=device,
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_special_tokens=n_special_tokens,
        n_segments=n_segments,
        enc_emb_dim=enc_emb_dim,
        enc_depth=enc_depth,
        pred_emb_dim=pred_emb_dim,
        pred_depth=pred_depth,
        pos_learnable=pos_learnable,
        seg_learnable=seg_learnable)
    target_encoder = copy.deepcopy(encoder)

    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)

    # Initialize mask collator
    if segment_masking:
       mask_collator = SegmentMaskCollator(
            n_targets=n_targets,
            n_contexts=n_contexts,
            seq_len_cell=seq_len_cell,
            seq_len_neighborhood=seq_len_neighborhood,
            n_special_tokens=n_special_tokens,
            per_segment_mask_ratio = per_segment_mask_ratio)
    else:
        mask_collator = MaskCollator(
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
        tokenizer_type=tokenizer_type,
        special_tokens=special_tokens,
        sampling_strategy=None)

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
    # ------------------------------------ #

    # ----------------------------- #
    #  Retrieve embeddings
    # ----------------------------- #
    target_encoder.eval()

    all_cell_ids = []
    all_cell_emb_list = []
    all_neighborhood_emb_list = []
    all_cell_gene_emb_dict = {}
    all_neighborhood_gene_emb_dict = {}

    for itr, (udata, masks_enc, masks_pred, masks_attention) in tqdm(enumerate(loader)):
        # Load gene tokens and segmentation label to the specified device
        tokens = udata[0].to(device, non_blocking=True)
        seg_label = udata[1].to(device, non_blocking=True)
        masks_attention = masks_attention.to(device, non_blocking=True)

        # Collect cell IDs to join metadata
        all_cell_ids.extend(udata[2])

        # Retrieve gene embeddings from different layers
        with torch.cuda.amp.autocast(dtype=torch.bfloat16,
                                     enabled=args['meta']['use_bfloat16']):

            emb_list = target_encoder.module.return_multi_layer_emb(
                tokens,
                seg_label,
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
                    selection_type='cls_cell',
                    seq_len_cell=seq_len_cell,
                    n_special_tokens=n_special_tokens)
                neighborhood_mask = create_binary_selection_mask(
                    tokens,
                    selection_type='cls_neighborhood',
                    seq_len_cell=seq_len_cell,
                    n_special_tokens=n_special_tokens)

                cell_emb = compute_mean_unmasked_emb(emb,
                                                     cell_mask)
                neighborhood_emb = compute_mean_unmasked_emb(
                    emb,
                    neighborhood_mask)
            # Keep elements relevant to cell embedding
            elif (agg_type == "avg") or (agg_type == "weighted_avg"):
                cell_mask = create_binary_selection_mask(
                    cell_neighborhood_tokens,
                    selection_type="agg_cell",
                    excluded_tokens=agg_excluded_tokens,
                    seq_len_cell=seq_len_cell,
                    n_special_tokens=n_special_tokens)
                neighborhood_mask = create_binary_selection_mask(
                    cell_neighborhood_tokens,
                    selection_type="agg_neighborhood",
                    excluded_tokens=agg_excluded_tokens,
                    seq_len_cell=seq_len_cell,
                    n_special_tokens=n_special_tokens)

                if agg_type == 'avg':
                    cell_emb = compute_mean_unmasked_emb(emb,
                                                         cell_mask)
                    neighborhood_emb = compute_mean_unmasked_emb(
                        emb,
                        neighborhood_mask)
                elif agg_type == "weighted_avg":
                    cell_weights = compute_unmasked_rank_based_weights(
                        cell_neighborhood_tokens, cell_mask)
                    cell_emb = compute_mean_unmasked_emb(
                        emb * cell_weights.unsqueeze(-1),
                        cell_mask)
                    neighborhood_weights = compute_unmasked_rank_based_weights(
                        cell_neighborhood_tokens, neighborhood_mask)
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
                for gene_id in cell_gene_ids:
                    gene_emb = retrieve_gene_emb(
                        tokens=cell_neighborhood_tokens,
                        emb=emb,
                        gene_id=gene_id,
                        gene_type="cell",
                        seq_len_cell=seq_len_cell,
                        n_special_tokens=n_special_tokens)
                    if itr == 0:
                        all_cell_gene_emb_dict[gene_id] = [gene_emb]
                    else:
                        all_cell_gene_emb_dict[gene_id].append(gene_emb)
                for gene_id in neighborhood_gene_ids:
                    gene_emb = retrieve_gene_emb(
                        tokens=cell_neighborhood_tokens,
                        emb=emb,
                        gene_id=gene_id,
                        gene_type="neighborhood",
                        seq_len_cell=seq_len_cell,
                        n_special_tokens=n_special_tokens)
                    if itr == 0:
                        all_neighborhood_gene_emb_dict[gene_id] = [gene_emb]
                    else:
                        all_neighborhood_gene_emb_dict[gene_id].append(gene_emb)                  

    adata = ad.AnnData(
        obs=pd.DataFrame({'cell_id': all_cell_ids},
        index=range(len(all_cell_ids))))

    # Add metadata
    adata_metadata = collect_adata_from_folder(raw_data_folder_path)
    merged_obs = pd.merge(adata.obs,
                         adata_metadata.obs,
                         on='cell_id')
    adata.obs = merged_obs.set_index('cell_id')
   
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

    return adata
