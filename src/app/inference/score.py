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

from app.utils import init_model, load_checkpoint
from terra.datasets.cell_datasets import CellBaseDataset, init_cell_dataset
from terra.datasets.dataloaders import init_dataloader_and_sampler
from terra.masks.block_masking  import BlockMaskCollator
from terra.masks.cell_masking import CellMaskCollator
from terra.tokenizers import cell_tokenizers
from terra.utils.embedding import (create_binary_selection_mask,
                                       compute_mean_unmasked_emb,
                                       compute_unmasked_rank_based_weights,
                                       collect_adata_from_folder,
                                       retrieve_gene_emb,
                                       compute_count_mean_cosine_sim,
                                       compute_sum_and_nonzero_count,
                                       batch_rowwise_distances)
from terra.utils.logging import CSVLogger
from typing import Dict, List


_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


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