import copy
import logging
import os
import pickle
import sys
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Literal, List

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from pyensembl import EnsemblRelease

from app.helper import init_model, load_checkpoint
from nichejepa.datasets.cell_datasets import CellBaseDataset, make_cell_dataset
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
                                       compute_sum_and_nonzero_count)

from nichejepa.utils.logging import CSVLogger


_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def find_differences(dict1, dict2):
    """
    Compare two dictionaries and return a list of keys where the values differ.
    Only keys present in both dictionaries are compared.
    """
    differing_keys = []
    for key in dict1:
        if key in dict2 and dict1[key][1].item() != dict2[key][1].item():
            differing_keys.append(key)
    return differing_keys

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
          compute_cosine_with: str='neighborhood',
          return_gene_per_data: bool=False,
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
    compute_cosine_with:
       If set to 'neighborhood', it will compute the cosine similarity between each cell and its neighborhood. 
       If set to 'cell', it will compute the cosine similarity between cells itself.
    return_gene_per_data:
        If 'True' will return gene_embedding for each gene per dataset.
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
        api_version=api_version)

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
    cell_dataset = make_cell_dataset(
        dataset=dataset,
        vocab_size=vocab_size,
        seq_len_cell=seq_len_cell,
        seq_len_neighborhood=seq_len_neighborhood,
        tokenizer_type=tokenizer_type,
        gt_type=gt_type,
        special_tokens=special_tokens,
        sampling_strategy=None,
        n_nonzero_tokens_list=n_nonzero_tokens)

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
    
    _, _, target_encoder, _, _, start_epoch = load_checkpoint(
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
    
    for itr, (udata, _, _, masks_attention) in tqdm(enumerate(loader)):
        # Load gene tokens and segmentation label to the specified device
        tokens = udata[0].to(device, non_blocking=True)
        segments = udata[1].to(device, non_blocking=True)
        if gt_type == 'rank':
            positions = udata[2].to(device, non_blocking=True)
        elif gt_type == 'counts':
            counts = udata[2].to(device, non_blocking=True)
        masks_attention = masks_attention.to(device, non_blocking=True)

        # Collect cell IDs to join metadata
        all_cell_ids.extend(udata[-1])

        # Aggregate gene embeddings into cell and neighborhood
        # embeddings
        ns_tokens = tokens[:, n_special_tokens:]

        # Exclude masked tokens from aggregation
        if masked_tokens is not None:
            mask_indices = torch.isin(
                tokens,
                torch.tensor(masked_tokens, device=tokens.device)
                ).unsqueeze(1).unsqueeze(1).expand(
                    -1,-1, tokens.shape[-1], -1)
            masks_attention[mask_indices] = 0

        # Retrieve gene embeddings from different layers
        with torch.cuda.amp.autocast(dtype=torch.bfloat16,
                                     enabled=args['meta']['use_bfloat16']):

            cell_emb_list = []
            neighborhood_emb_list = []
            for emb_layer in emb_layers:
                if gt_type == 'rank':
                    cell_emb_list.append(return_layer_emb_fn(
                        layer=emb_layer,
                        positions=positions,
                        segments=segments,
                        tokens=tokens,
                        masks_attention=masks_attention,
                        pad_neighborhood=True).cpu()),
                    neighborhood_emb_list.append(return_layer_emb_fn(
                        layer=emb_layer,
                        positions=positions,
                        segments=segments,
                        tokens=tokens,
                        masks_attention=masks_attention,
                        pad_neighborhood=False).cpu()),
                elif gt_type == 'counts':
                    cell_emb_list.append(return_layer_emb_fn(
                        layer=emb_layer,
                        tokens=tokens,
                        segments=segments,
                        counts=counts,
                        masks_attention=masks_attention,
                        pad_neighborhood=True).cpu()),
                    neighborhood_emb_list.append(return_layer_emb_fn(
                        layer=emb_layer,
                        tokens=tokens,
                        segments=segments,
                        counts=counts,
                        masks_attention=masks_attention,
                        pad_neighborhood=False).cpu()),
        
            if feature_norm and (emb_layers[-1] == enc_depth):
                # Normalize last layer like in training
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
                emb = n_emb
                if len(cell_gene_ids) != 0 or len(neighborhood_gene_ids) != 0 :
                    if itr == 0 or itr == len(loader)-1:
                        cell_embs = torch.zeros((emb.shape[0], len(cell_gene_ids), emb.shape[-1]), device=emb.device)
                        cell_presence = torch.zeros((emb.shape[0], len(cell_gene_ids)), device=emb.device)
                        neb_occ_list: List[torch.Tensor] = []
                        neb_occ_mask_list: List[torch.Tensor] = []
                        neb_presence = torch.zeros((emb.shape[0], len(neighborhood_gene_ids)), device=emb.device)
                    else:
                        cell_embs.zero_()
                        cell_presence.zero_()
                        neb_occ_list = []
                        neb_occ_mask_list = []
                        neb_presence.zero_()
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
                            all_cell_gene_emb_dict[gene_id] = [cell_embs[:, j, :]]
                        else:
                            all_cell_gene_emb_dict[gene_id].append(cell_embs[:, j, :])
                    if return_gene_per_data:
                        gene_sum, gene_count = compute_sum_and_nonzero_count(cell_embs[:, j, :])
                        if itr == 0:
                            all_cell_gene_emb_per_data_dict[gene_id] = (gene_sum, gene_count)
                        else:
                            all_cell_gene_emb_per_data_dict[gene_id][0].add_(gene_sum)
                            all_cell_gene_emb_per_data_dict[gene_id][1].add_(gene_count)
                # Process neighborhood genes (multiple occurrences: compute cosine per occurrence)
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
                    if return_gene:
                        if itr == 0:
                            #all_neighborhood_gene_emb_dict[gene_id] = [gene_occ * occ_mask.unsqueeze(-1)]
                            all_neighborhood_gene_emb_dict[gene_id] = [compute_mean_unmasked_emb(gene_occ,occ_mask)]

                        else:
                            #all_neighborhood_gene_emb_dict[gene_id].append(gene_occ * occ_mask.unsqueeze(-1))
                            all_neighborhood_gene_emb_dict[gene_id].append(compute_mean_unmasked_emb(gene_occ,occ_mask))
                    if return_gene_per_data:
                        gene_sum, gene_count = compute_sum_and_nonzero_count(compute_mean_unmasked_emb(gene_occ,occ_mask))
                        if itr == 0:
                            all_neighborhood_gene_emb_per_data_dict[gene_id] = (gene_sum, gene_count)
                        else:
                            all_neighborhood_gene_emb_per_data_dict[gene_id][0].add_(gene_sum)
                            all_neighborhood_gene_emb_per_data_dict[gene_id][1].add_(gene_count)
                # Stack neighborhood gene occurrence tensors along gene dimension:
                # Resulting shape: (N, num_neb_genes, max_occ, D) and mask: (N, num_neb_genes, max_occ)
                if len(neighborhood_gene_ids) != 0 and return_cosine_sim:
                    neb_occ_tensor = torch.stack(neb_occ_list, dim=1)
                    neb_occ_mask_tensor = torch.stack(neb_occ_mask_list, dim=1)
                # Compute cosine similarity components using our function for multiple occurrences.
                if return_cosine_sim:
                    if itr == 0:
                        sum_cos_sim, pair_count, cell_count = compute_count_mean_cosine_sim(cell_embs, cell_presence, neb_occ_tensor, neb_occ_mask_tensor)
                    else:
                        sum_cos_sim_temp, pair_count_temp, cell_count_temp = compute_count_mean_cosine_sim(cell_embs, cell_presence, neb_occ_tensor, neb_occ_mask_tensor)
                        sum_cos_sim.add_(sum_cos_sim_temp)
                        pair_count.add_(pair_count_temp)
                        cell_count.add_(cell_count_temp)
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
        for gene_id in neighborhood_gene_ids:
            adata.obsm[f"neighborhood_emb_gene{gene_id}"] = np.array(torch.cat(
                all_neighborhood_gene_emb_dict[gene_id],
                dim=0).cpu())
    if return_cosine_sim:
        adata.uns['sum_cos_sim'] = sum_cos_sim.numpy()
        adata.uns['cos_sim_pair_count'] = pair_count.numpy()
        adata.uns['cos_sim_cell_count'] = cell_count.numpy()
        adata.uns['cos_sim'] = adata.uns['sum_cos_sim'] / adata.uns['cos_sim_pair_count']
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
                cell_gene_emb_features.append(sum_emb / count_emb)
                cell_gene_emb_counts.append(count_emb)

        for gene_id in neighborhood_gene_ids:
            if gene_id in all_neighborhood_gene_emb_per_data_dict.keys():
                sum_emb = all_neighborhood_gene_emb_per_data_dict[gene_id][0].numpy()
                count_emb = all_neighborhood_gene_emb_per_data_dict[gene_id][1].numpy()
                neighborhood_gene_emb_features.append(sum_emb / count_emb)
                neighborhood_gene_emb_counts.append(count_emb)

        # Concatenate all features, sums, and counts into single numpy arrays
        adata.uns['cell_gene_emb_average_per_data'] = np.concatenate(cell_gene_emb_features, axis=0)
        adata.uns['cell_gene_emb_counts_per_data'] = np.concatenate(cell_gene_emb_counts, axis=0)

        adata.uns['neighborhood_gene_emb_average_per_data'] = np.concatenate(neighborhood_gene_emb_features, axis=0)
        adata.uns['neighborhood_gene_emb_counts_per_data'] = np.concatenate(neighborhood_gene_emb_counts, axis=0)
    if len(cell_gene_ids) != 0:
        adata.uns['cell_gene_ids']=np.array(cell_gene_ids)
    if len(neighborhood_gene_ids) != 0:
        adata.uns['neighborhood_gene_ids']=np.array(neighborhood_gene_ids)

    return adata


def harmonize_adata(adata: ad.AnnData) -> ad.AnnData:
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
    print('Adding ensembl IDs from release 111...')
    # Extract ensembl IDs of protein coding and miRNA mouse genes
    ensembl = EnsemblRelease(release=111, species='human')
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
                   perturb_df: pd.DataFrame | None = None,                   
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
    perturb_df:
        DataFrame with perturbation data, e.g.
        ```
        perturb_df =  pd.DataFrame({
            'ensembl_id': ['ENSG00000169194', 'ENSG00000131724'],
            'target': ['neighborhood', 'cell'],
            'perturbation_type': ['foldchange', 'knockout'],
            'foldchange': [0.5, np.nan]
        })
        ```.
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
    if not isinstance(perturb_df, pd.DataFrame):
        if perturb_df is None:
            print('STEP 2: TOKENIZING ANNDATA OBJECT...')
        else:
            raise ValueError('`perturb_df` must be a pd.DataFrame.')
    else:
        print('STEP 2: TOKENIZING ANNDATA OBJECT AND APPLYING PERTURBATIONS...')
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
    dataset_dict = tk._tokenize_adata(adata=adata,
                                      perturb_df=perturb_df)
    dataset = tk._create_dataset(
        dataset_dict=dataset_dict,
        use_generator=False,
        cache_directory_path=None,
        keep_in_memory=False)
    
    return dataset


@torch.no_grad()
def embed_dataset(dataset: Dataset,
                  model_folder_path: str,
                  emb_layer: int | None = None,
                  cell_gene_ids: list = [],
                  neighborhood_gene_ids: list = [],
                  agg_excluded_tokens: list[int] | None = None,
                  top_k: int | None = None,
                  batch_size: int = 128,
                  pin_memory: bool = False,
                  num_workers: int = 12
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
        use_flash_attention=model_config['meta']['use_flash_attention'])

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
    cell_dataset = make_cell_dataset(
        dataset=dataset,
        vocab_size=vocab_size,
        seq_len_cell=model_config['data']['seq_len_cell'],
        seq_len_neighborhood=model_config['data']['seq_len_neighborhood'],
        max_special_tokens=max_special_tokens,
        tokenizer_type=model_config['data']['tokenizer_type'],
        gt_type=model_config['meta']['gt_type'],
        special_tokens=model_config['meta']['special_tokens'],
        sampling_strategy=None,
        n_nonzero_tokens_list=None)

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
    _, _, target_encoder, _, _, start_epoch = load_checkpoint(
            device=device,
            r_path=model_checkpoint_path,
            encoder=None,
            predictor=None,
            target_encoder=target_encoder,
            opt=None,
            scaler=None)
    target_encoder.eval()

    # Retrieve embeddings
    all_cell_emb_list = []
    all_neighborhood_emb_list = []
    all_cell_gene_emb_dict = {}
    all_neighborhood_gene_emb_dict = {}

    for itr, (udata, _, _, masks_attention) in tqdm(enumerate(loader)):
        # Load gene tokens and segmentation label to the specified device
        tokens = udata[0].to(device, non_blocking=True)
        segments = udata[1].to(device, non_blocking=True)
        if model_config['meta']['gt_type'] == 'rank':
            positions = udata[2].to(device, non_blocking=True)
        elif model_config['meta']['gt_type'] == 'counts':
            counts = udata[2].to(device, non_blocking=True)
        masks_attention = masks_attention.to(device, non_blocking=True)

        # Aggregate gene embeddings into cell and neighborhood embeddings
        ns_tokens = tokens[:, n_special_tokens:]

        # Retrieve gene embeddings from different layers
        with torch.cuda.amp.autocast(
            dtype=torch.bfloat16,
            enabled=model_config['meta']['use_bfloat16']):

            if model_config['meta']['gt_type'] == 'rank':
                emb = target_encoder.return_layer_emb(
                    layer=emb_layer,
                    positions=positions,
                    segments=segments,
                    tokens=tokens,
                    masks_attention=masks_attention).cpu()
            elif model_config['meta']['gt_type'] == 'counts':
                emb = target_encoder.return_layer_emb(
                    layer=emb_layer,
                    tokens=tokens,
                    segments=segments,
                    counts=counts,
                    masks_attention=masks_attention).cpu()
        
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
        cell_emb = compute_mean_unmasked_emb(emb, cell_mask)
        neighborhood_emb = compute_mean_unmasked_emb(emb, neighborhood_mask)

        all_cell_emb_list.append(cell_emb)
        all_neighborhood_emb_list.append(neighborhood_emb)

        # Store cell and neighborhood gene embeddings
        for gene_id in cell_gene_ids:
            gene_emb = retrieve_gene_emb(
                tokens=tokens,
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
                tokens=tokens,
                emb=emb,
                gene_id=gene_id,
                gene_type="neighborhood",
                seq_len_cell=seq_len_cell,
                n_special_tokens=n_special_tokens)
            if itr == 0:
                all_neighborhood_gene_emb_dict[gene_id] = [gene_emb]
            else:
                all_neighborhood_gene_emb_dict[gene_id].append(gene_emb)

    output_embed = {}        

    # Store cell and neighborhood embeddings of all observations
    output_embed["cell_emb"] = np.array(torch.cat(
        all_cell_emb_list,
        dim=0))
    output_embed["neighborhood_emb"] = np.array(torch.cat(
        all_neighborhood_emb_list,
        dim=0))

    # Store cell and neighborhood gene embeddings of all observations
    for gene_id in cell_gene_ids:
        output_embed[f"cell_emb_gene{gene_id}"] = np.array(torch.cat(
            all_cell_gene_emb_dict[gene_id],
            dim=0).cpu())
    for gene_id in neighborhood_gene_ids:
        output_embed[f"neighborhood_emb_gene{gene_id}"] = np.array(torch.cat(
            all_neighborhood_gene_emb_dict[gene_id],
            dim=0).cpu())

    return output_embed


@torch.no_grad()
def harmonize_tokenize_embed_pipeline(
        adata: ad.AnnData,
        model_folder_path: str,
        perturb_df: pd.DataFrame | None = None,               
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
    perturb_df:
        DataFrame with perturbation data, e.g.
        ```
        perturb_df =  pd.DataFrame({
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
        perturb_df=perturb_df,                   
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
