import copy
import logging
import os
import sys
import yaml
from typing import Literal

import anndata
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from .datasets.cell_neighborhood_dataset import (CellNeighborhoodDataset,
                                                 make_cell_neighborhood_dataset)
from .helper import init_model, load_checkpoint
from .masks.multigene import MaskCollator
from .masks.segment_masking  import SegmentMaskCollator
from .utils.distributed import init_distributed
from .utils.embedding import (create_binary_selection_mask,
                              compute_mean_unmasked_emb,
                              compute_unmasked_rank_based_weights,
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
          dataset: CellNeighborhoodDataset,
          cell_gene_ids: list=[],
          neighborhood_gene_ids: list=[],
          agg_type: Literal['cls', 'avg', 'weighted_avg']='avg',
          feature_norm: bool=False,
          check_point: int=None
          ) -> anndata.AnnData:
    """
    Use a trained model for inference. Run forward pass on a given dataset and
    return cell, neighborhood and (optionally) gene embeddings (cell and
    neighborhood gene embeddings).

    Parameters
    -----------
    args:
        Dictionary containing the hyperparameters from the config file.
    dataset:
        CellNeighborhoodDataset for which embeddings will be inferred.
    cell_gene_ids:
        List with gene IDs for which cell gene embeddings will be retrived.
    neighborhood_gene_ids:
        List with gene IDs for which neighborhood gene embeddings will be
        retrived.
    agg_type:
        Specifies how (aggregated) cell and neighborhood embeddings are computed
        from individual gene embeddings.
    feature_norm:
        If 'True', apply feature norm in the last embedding layer.
    check_point:
        If a checkpoint is provided, the model will load from it; 
        otherwise, it will load the latest saved model.
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
    pos_learnable = args['meta']['pos_learnable']
    seg_learnable = args['meta']['seg_learnable']

    # Load data params
    batch_size = args['data']['batch_size']
    vocab_size = args['data']['vocab_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    data_set_name = args['data']['data_set_name']
    seq_len_cell = args['data']['seq_len_cell']
    seq_len_neighborhood = args['data']['seq_len_neighborhood']
    has_cls = args['data']['has_cls']

    # Load optimization params
    num_epochs = args['optimization']['epochs']
    if isinstance(args['optimization']['ema'], list):
       ema = args['optimization']['ema']
    else:
       ema = [args['optimization']['ema'], 1]

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
    
    # Compute seq_len based on config
    seq_len = seq_len_cell + seq_len_neighborhood

    # Set the folder for saving extracted features
    folder = (f"logs/{data_set_name}_"
              f"pred_depth_{pred_depth}_pred_emb_dim_{pred_emb_dim}_"
              f"enc_depth_{enc_depth}_enc_emb_dim_{enc_emb_dim}_n_targets_{n_targets}_"
              f"n_contexts_{n_contexts}_target_mask_size_{target_mask_size}_"
              f"ema_{ema[0]:.4f}_num_epochs_{num_epochs}_"
              f"seq_len_cell_{seq_len_cell}_"
              f"seq_len_neighborhood_{seq_len_neighborhood}_"
              f"pos_learnable_{pos_learnable}_"
              f"seg_learnable_{seg_learnable}_"
              f"ratio_{per_segment_mask_ratio:.4f}")
    save_folder = f"{folder}/extracted_features"
    feature_path = f"{save_folder}/"

    os.makedirs(save_folder, exist_ok=True)
    tag = args['logging']['write_tag']
    dump = os.path.join(folder, f'params.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)

    # Define checkpointing path
    if check_point is None:
       latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    else:
       latest_path = os.path.join(folder, f'{tag}-ep{check_point}.pth.tar')
    load_path = (os.path.join(folder, r_file) if r_file is not None else
        latest_path)

    # Initialize encoder, predictor, and target encoder
    encoder, predictor = init_model(
        device=device,
        vocab_size=vocab_size,
        seq_len=seq_len,
        enc_emb_dim=enc_emb_dim,
        enc_depth=enc_depth,
        pred_emb_dim=pred_emb_dim,
        pred_depth=pred_depth,
        pos_learnable=pos_learnable,
        seg_learnable=seg_learnable,
        has_cls=has_cls)
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
            has_cls=has_cls,
            per_segment_mask_ratio = per_segment_mask_ratio)
    else:
        mask_collator = MaskCollator(
            n_targets=n_targets,
            n_contexts=n_contexts,
            target_mask_size=target_mask_size,
            context_mask_size=context_mask_size,
            seq_len_cell=seq_len_cell,
            seq_len_neighborhood=seq_len_neighborhood,
            has_cls=has_cls)

    # Initialize dataloader
    _, loader = make_cell_neighborhood_dataset(
        batch_size=batch_size,
        data=dataset,
        vocab_size=vocab_size,
        collator=mask_collator,
        pin_mem=pin_mem,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        drop_last=False,
        seq_len_cell=seq_len_cell,
        seq_len_neighborhood=seq_len_neighborhood,
        has_cls=has_cls,
        distributed=False)
    
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
    niche_label = []
    cell_type_label = []
    all_cell_emb_list = []
    all_neighborhood_emb_list = []
    all_cell_gene_emb_dict = {}
    all_neighborhood_gene_emb_dict = {}

    for itr, (udata, masks_enc, masks_pred, masks_attention) in tqdm(enumerate(loader)):
        # Load cell neighborhood tokens and segmentation label to the specified
        # device
        cell_neighborhood_tokens = udata[0].to(device, non_blocking=True)
        seg_label = udata[1].to(device, non_blocking=True)
        masks_attention = masks_attention.to(device, non_blocking=True)

        # Load niche and cell type labels based on specified sequence lengths
        if (args['data']['seq_len_cell'] > 0) & (
            args['data']['seq_len_neighborhood'] > 0):
            cell_type_label.extend(udata[2])
            niche_label.extend(udata[3])
        elif args['data']['seq_len_cell'] > 0:
            cell_type_label.extend(udata[2])
        elif args['data']['seq_len_neighborhood'] > 0:
            niche_label.extend(udata[2])

        # Retrieve gene embeddings from different layers
        with torch.cuda.amp.autocast(dtype=torch.bfloat16,
                                     enabled=args['meta']['use_bfloat16']):
            emb_list = target_encoder.module.return_multi_layer_emb(
                cell_neighborhood_tokens, seg_label, masks_attention=masks_attention)
        
            if feature_norm:
                # Normalize last layer like in training
                emb_list[-1] = F.layer_norm(emb_list[-1],
                                            (emb_list[-1].size(-1),))

        # Aggregate gene embeddings into cell and neighborhood embeddings
        for i, emb in enumerate(emb_list):
            # Keep only <cls> token; at the moment there is only 1 <cls> token
            if agg_type == "cls":
                cell_mask = create_binary_selection_mask(
                    cell_neighborhood_tokens,
                    selection_type=agg_type,
                    seq_len_cell=seq_len_cell,
                    has_cls=has_cls)
                neighborhood_mask = create_binary_selection_mask(
                    cell_neighborhood_tokens,
                    selection_type=agg_type,
                    seq_len_cell=seq_len_cell,
                    has_cls=has_cls)
            # Keep elements relevant to cell embedding
            elif (agg_type == "avg") or (agg_type == "weighted_avg"):
                cell_mask = create_binary_selection_mask(
                    cell_neighborhood_tokens,
                    selection_type="agg_cell",
                    seq_len_cell=seq_len_cell,
                    has_cls=has_cls)
                neighborhood_mask = create_binary_selection_mask(
                    cell_neighborhood_tokens,
                    selection_type="agg_neighborhood",
                    seq_len_cell=seq_len_cell,
                    has_cls=has_cls)

                if agg_type == 'avg':
                    cell_emb = compute_mean_unmasked_emb(emb,
                                                         cell_mask)
                    neighborhood_emb = compute_mean_unmasked_emb(
                        emb,
                        neighborhood_mask)
                elif agg_type == "weighted_avg":
                    cell_weights = compute_unmasked_rank_based_weights(
                        cell_neighborhood_tokens, cell_mask)
                    cell_emb, _ = compute_mean_unmasked_emb(
                        emb * cell_weights.unsqueeze(-1),
                        cell_mask)
                    neighborhood_weights = compute_unmasked_rank_based_weights(
                        cell_neighborhood_tokens, neighborhood_mask)
                    neighborhood_emb, _ = compute_mean_unmasked_emb(
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
                        has_cls=has_cls,
                        seq_len_cell=seq_len_cell)
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
                        has_cls=has_cls,
                        seq_len_cell=seq_len_cell)
                    if itr == 0:
                        all_neighborhood_gene_emb_dict[gene_id] = [gene_emb]
                    else:
                        all_neighborhood_gene_emb_dict[gene_id].append(gene_emb)                  
                    
    adata = anndata.AnnData(
        obs=pd.DataFrame({
            'niche': niche_label,
            'cell_type': cell_type_label},
        index=range(len(niche_label))))

    # Store cell and neighborhood embeddings of all observations across layers  
    for i in range(len(all_cell_emb_list)):
        print(np.array(torch.cat(
            all_cell_emb_list[i],
            dim=0).cpu()).shape)
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
