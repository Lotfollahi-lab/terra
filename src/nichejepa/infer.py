import copy
import logging
import os
import pickle
import sys
import yaml
from collections import defaultdict
from pathlib import Path
from typing import List, Literal, Optional

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
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
          dataset_ids: list | None = None,
          obs_cols: list | None = None,
          uns_cols: list | None = None,
          cell_gene_ids: List=[],
          neighborhood_gene_ids: List=[],
          agg_type: Literal['cls_cell',
                            'cls_neighborhood',
                            'avg',
                            'weighted_avg']='avg',
          masked_tokens: Optional[List[int]]=None,
          agg_excluded_tokens: Optional[List[int]]=None,
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
    masked_tokens:
        List of tokens to be masked by the attention mask during inference.
    agg_excluded_tokens:
        List of tokens to be excluded from the aggregation.
    feature_norm:
        If 'True', apply feature norm in the last embedding layer.

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
    if 'new_spc' in args['meta'].keys():
        new_spc = args['meta']['new_spc']
    else:
        new_spc = False

    dataset_name = args['data']['dataset_name']
    raw_data_folder_path = args['data']['raw_data_folder_path']
    batch_size = args['data']['batch_size']
    vocab_size = args['data']['vocab_size']
    pin_memory = args['data']['pin_memory']
    num_workers = args['data']['num_workers']
    tokenizer_type = args['data']['tokenizer_type']
    n_special_values = args['data']['n_special_values']
    seq_len_cell = args['data']['seq_len_cell']
    seq_len_neighborhood = args['data']['seq_len_neighborhood']
    n_segments = args['data']['n_segments']

    n_contexts = args['mask']['n_contexts']
    n_targets = args['mask']['n_targets']
    block_masking = args['mask']['block_masking']
    context_mask_size = args['mask']['context_mask_size']
    target_mask_size = args['mask']['target_mask_size']
    per_block_mask_ratio = args['mask']['per_block_mask_ratio']
    if 'restrict_special_attention' in args['meta'].keys():
        restrict_special_attention = args['meta']['restrict_special_attention']
    else:
        restrict_special_attention = False

    r_file = args['state']['read_checkpoint']
    tag = args['state']['write_tag']
    
    # Define tokenizer-specific params
    if tokenizer_type == 'cell_neighborhood':
        max_special_tokens = 7
        max_cls_tokens = args['meta']['n_cls']
        special_tokens = ['cls_cell', 'cls_neighborhood'] + special_tokens
    elif tokenizer_type == 'cell_graph':
        max_special_tokens = 105
        max_cls_tokens = args['meta']['n_cls']
        special_tokens = [
            f'cls_{i}' for i in range(max_cls_tokens)] + special_tokens

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
    latest_path = os.path.join(load_folder_path, f'{tag}-latest.pth.tar')
    load_path = (os.path.join(load_folder_path, r_file) if r_file is not None 
        else latest_path)

    # Initialize encoder, predictor, and target encoder
    target_encoder, _ = init_model(
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
        new_spc=new_spc)

    # Initialize mask collator
    if block_masking:
       mask_collator = BlockMaskCollator(
            n_targets=n_targets,
            n_contexts=n_contexts,
            n_segments=n_segments,
            seq_len_cell=seq_len_cell,
            seq_len_neighborhood=seq_len_neighborhood,
            max_special_tokens=max_special_tokens,
            n_special_tokens=n_special_tokens,
            max_cls_tokens=max_cls_tokens,
            per_block_mask_ratio=per_block_mask_ratio,
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
        include_cell_id=True)

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
    
    _, _, target_encoder, _, _, start_epoch, _ = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=None,
            predictor=None,
            target_encoder=target_encoder,
            opt=None,
            scaler=None,
            is_training=False)
   
    # Retrieve embeddings
    target_encoder.eval()

    all_cell_ids = []
    all_cell_emb_list = []
    all_neighborhood_emb_list = []
    all_cell_gene_emb_dict = {}
    all_neighborhood_gene_emb_dict = {}

    for itr, (udata, _, _, masks_attention) in tqdm(enumerate(loader)):
        # Load gene tokens and segmentation label to the specified device
        for key, val in udata.items():
            if key != 'cell_id':
                udata[key] = val.to(device, non_blocking=True)
        masks_attention = masks_attention.to(device, non_blocking=True)

        # Collect cell IDs to join metadata
        all_cell_ids.extend(udata['cell_id'])

        # Retrieve gene embeddings from different layers
        with torch.cuda.amp.autocast(dtype=torch.bfloat16,
                                     enabled=args['meta']['use_bfloat16']):

            if masked_tokens is not None:
                mask_indices = torch.isin(
                    udata["tokens"],
                    torch.tensor(masked_tokens, device=udata["tokens"].device)
                    ).unsqueeze(1).unsqueeze(1).expand(-1, -1, 1108, -1) # temp
                masks_attention[mask_indices] = 0

            emb_list = target_encoder.backbone.return_multi_layer_emb(
                batch=udata,
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
                    udata["tokens"],
                    selection_type='cls_0',
                    seq_len_cell=seq_len_cell,
                    n_special_tokens=n_special_tokens,
                    max_cls_tokens=max_cls_tokens)
                if tokenizer_type == 'cell_neighborhood':
                    neighborhood_mask = create_binary_selection_mask(
                        udata["tokens"],
                        selection_type='cls_1',
                        seq_len_cell=seq_len_cell,
                        n_special_tokens=n_special_tokens,
                        max_cls_tokens=max_cls_tokens)
                elif tokenizer_type == 'cell_graph':
                    neighborhood_mask = create_binary_selection_mask(
                        udata["tokens"],
                        selection_type='cls_all',
                        seq_len_cell=seq_len_cell,
                        n_special_tokens=n_special_tokens,
                        max_cls_tokens=max_cls_tokens)

                cell_emb = compute_mean_unmasked_emb(emb,
                                                     cell_mask)
                neighborhood_emb = compute_mean_unmasked_emb(
                    emb,
                    neighborhood_mask)
            # Keep elements relevant to cell embedding
            elif (agg_type == "avg") or (agg_type == "weighted_avg"):
                cell_mask = create_binary_selection_mask(
                    udata["tokens"],
                    selection_type="agg_cell",
                    excluded_tokens=agg_excluded_tokens,
                    seq_len_cell=seq_len_cell,
                    n_special_tokens=n_special_tokens,
                    max_cls_tokens=max_cls_tokens)
                neighborhood_mask = create_binary_selection_mask(
                    udata["tokens"],
                    selection_type="agg_neighborhood",
                    excluded_tokens=agg_excluded_tokens,
                    seq_len_cell=seq_len_cell,
                    n_special_tokens=n_special_tokens,
                    max_cls_tokens=max_cls_tokens)

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
                for gene_id in cell_gene_ids:
                    gene_emb = retrieve_gene_emb(
                        tokens=udata["tokens"],
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
                        tokens=udata["tokens"],
                        emb=emb,
                        gene_id=gene_id,
                        gene_type="neighborhood",
                        seq_len_cell=seq_len_cell,
                        n_special_tokens=n_special_tokens)
                    if itr == 0:
                        all_neighborhood_gene_emb_dict[gene_id] = [gene_emb]
                    else:
                        all_neighborhood_gene_emb_dict[gene_id].append(gene_emb)                  

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


@torch.inference_mode()
def embed_dataset(dataset: Dataset,
                  model_folder_path: str,
                  emb_layer: int | None = None,
                  agg_excluded_tokens: list[int] | None = None,
                  top_k: int | None = None,
                  batch_size: int = 128,
                  pin_memory: bool = False,
                  num_workers: int = 12,
                  include_spatial_cell_emb: bool = False,
                  return_token_embeddings: bool = False,
                  masked_tokens = None,
                  feature_norm: bool=False,
                  agg_type: str='cls',
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

    # Define tokenizer-specific params
    if model_config['data']['tokenizer_type'] == 'cell_neighborhood':
        max_special_tokens = 7
        max_cls_tokens = model_config['meta']['n_cls']
        special_tokens = ['cls_cell', 'cls_neighborhood'] + model_config['meta']['special_tokens']
    elif model_config['data']['tokenizer_type'] == 'cell_graph':
        max_special_tokens = 105
        max_cls_tokens = model_config['meta']['n_cls']
        special_tokens = [
            f'cls_{i}' for i in range(max_cls_tokens)] + model_config['meta']['special_tokens']

    # Get token sequence length and number of special tokens
    n_special_tokens = len(special_tokens)
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
    n_special_values = model_config['data']['n_special_values']
    #n_special_values = sum(1 for key in token_dict if "spv" in key)

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
        device=device,
        vocab_size=vocab_size,
        seq_len=seq_len,
        max_cls_tokens=max_cls_tokens,
        max_special_tokens=max_special_tokens,
        n_special_tokens=n_special_tokens,
        n_segments=model_config['data']['n_segments'],
        n_special_values=n_special_values,
        enc_emb_dim=model_config['meta']['enc_emb_dim'],
        enc_depth=model_config['meta']['enc_depth'],
        pred_emb_dim=model_config['meta']['pred_emb_dim'],
        pred_depth=model_config['meta']['pred_depth'],
        pos_learnable=model_config['meta']['pos_learnable'],
        seg_learnable=model_config['meta']['seg_learnable'],
        new_spc=model_config['meta']['new_spc'])

    # Create mask collator
    mask_collator = BlockMaskCollator(
        n_targets=model_config['mask']['n_targets'],
        n_contexts=model_config['mask']['n_contexts'],
        n_segments=model_config['data']['n_segments'],
        seq_len_cell=model_config['data']['seq_len_cell'],
        seq_len_neighborhood=model_config['data']['seq_len_neighborhood'],
        max_special_tokens=max_special_tokens,
        n_special_tokens=n_special_tokens,
        max_cls_tokens=max_cls_tokens,
        per_block_mask_ratio=model_config['mask']['per_block_mask_ratio'],
        restrict_special_attention=model_config['mask']['restrict_special_attention'],
        sample_segments=False)
        
    # Create torch dataset
    cell_dataset = make_cell_dataset(
        dataset=dataset,
        vocab_size=vocab_size,
        seq_len_cell=model_config['data']['seq_len_cell'],
        seq_len_neighborhood=model_config['data']['seq_len_neighborhood'],
        max_cls_tokens=max_cls_tokens,
        max_special_tokens=max_special_tokens,
        tokenizer_type=model_config['data']['tokenizer_type'],
        gt_type=model_config['meta']['gt_type'],
        special_tokens=special_tokens,
        sampling_strategy=None,
        include_cell_id=True)

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
    if return_token_embeddings:
        all_token_emb_list = []

    for itr, (udata, _, _, masks_attention) in tqdm(enumerate(loader)):
        for key in udata.keys():
            if key != 'cell_id':
                udata[key] = udata[key].to(device, non_blocking=True)
        masks_attention = masks_attention.to(device, non_blocking=True)

        # Retrieve gene embeddings from different layers
        with torch.cuda.amp.autocast(dtype=torch.bfloat16,
                                     enabled=model_config['meta']['use_bfloat16']):

            if masked_tokens is not None:
                mask_indices = torch.isin(
                    udata["tokens"],
                    torch.tensor(masked_tokens, device=udata["tokens"].device)
                    ).unsqueeze(1).unsqueeze(1).expand(-1, -1, 1108, -1) # temp
                masks_attention[mask_indices] = 0

            emb_list = target_encoder.backbone.return_multi_layer_emb(
                batch=udata,
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
                    udata["tokens"],
                    selection_type='cls_0',
                    seq_len_cell=model_config['data']['seq_len_cell'],
                    n_special_tokens=n_special_tokens,
                    max_cls_tokens=max_cls_tokens)
                if model_config['data']['tokenizer_type'] == 'cell_neighborhood':
                    neighborhood_mask = create_binary_selection_mask(
                        udata["tokens"],
                        selection_type='cls_1',
                        seq_len_cell=model_config['data']['seq_len_cell'],
                        n_special_tokens=n_special_tokens,
                        max_cls_tokens=max_cls_tokens)
                elif model_config['data']['tokenizer_type'] == 'cell_graph':
                    neighborhood_mask = create_binary_selection_mask(
                        udata["tokens"],
                        selection_type='cls_all',
                        seq_len_cell=model_config['data']['seq_len_cell'],
                        n_special_tokens=n_special_tokens,
                        max_cls_tokens=max_cls_tokens)

                cell_emb = compute_mean_unmasked_emb(emb,
                                                     cell_mask)
                neighborhood_emb = compute_mean_unmasked_emb(
                    emb,
                    neighborhood_mask)
            # Keep elements relevant to cell embedding
            elif (agg_type == "avg") or (agg_type == "weighted_avg"):
                cell_mask = create_binary_selection_mask(
                    udata["tokens"],
                    selection_type="agg_cell",
                    excluded_tokens=agg_excluded_tokens,
                    seq_len_cell=model_config['data']['seq_len_cell'],
                    n_special_tokens=n_special_tokens,
                    max_cls_tokens=max_cls_tokens)
                neighborhood_mask = create_binary_selection_mask(
                    udata["tokens"],
                    selection_type="agg_neighborhood",
                    excluded_tokens=agg_excluded_tokens,
                    seq_len_cell=model_config['data']['seq_len_cell'],
                    n_special_tokens=n_special_tokens,
                    max_cls_tokens=max_cls_tokens)

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