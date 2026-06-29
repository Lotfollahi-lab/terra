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

from terra.utils.helper import init_model, load_checkpoint
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


logger = logging.getLogger(__name__)


def tokenize_adata(adata: ad.AnnData,
                   model_folder_path: str,
                   cache_directory_path: str,             
                   nproc: int = 4,
                   processing_mode: Literal['sequential',
                                            'parallel'] = 'parallel',
                   add_neigh_cell_ids: bool = False,
                   use_generator: bool = True,
                   keep_in_memory: bool = False,
                   include_special_tokens: bool = False,
                   ) -> Dataset:
    """
    Tokenize an AnnData object based on the parameters in the model config and
    return the tokenized Hugging Face dataset.

    Parameters
    -----------
    adata:
        AnnData object to be tokenized.
    model_folder_path:
        Path to the folder containing the model config, token dictionary, and
        normalization factors.
    cache_directory_path:
        Path where the cache is stored during dataset creation.
    nproc:
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
    include_special_tokens:
        Whether to include special tokens in the tokenized output.

    Returns
    -----------
    dataset : datasets.Dataset
        The tokenized data stored in a Hugging Face dataset.
    """
    logger.info('STEP 1: LOADING CONFIG...')
    model_config_file_path = Path(model_folder_path) / 'model_config.yaml'
    token_dictionary_file_path = Path(model_folder_path) / 'token_dictionary.pkl'

    # Load model config
    with open(model_config_file_path, 'r') as file:
        model_config = yaml.safe_load(file)

    # Normalization-artifact paths are read FROM THE MODEL CONFIG (recorded at
    # training time under config['data']). If a path is specified, it is used
    # and logged; if it is absent, it is not read and a warning notes the
    # per-file fallback. This keeps the inference normalization explicit and
    # matched to how the model was trained -- no silent mismatch.
    norm_factor_file_path = model_config['data'].get('norm_factor_file_path')
    pf_targets_file_path = model_config['data'].get('pf_targets_file_path')

    if norm_factor_file_path:
        logger.info("Gene norm factors: reading from config path "
              f"'{norm_factor_file_path}'.")
    else:
        logger.warning("'norm_factor_file_path' is not set in the model "
              "config['data'] -> gene-level norm factors will NOT be loaded. "
              "This is correct only if the model uses no gene-level norm "
              "method (e.g. shifted_log / pflog1ppf).")

    if pf_targets_file_path:
        logger.info("PFlog1pPF: reading FROZEN corpus targets from config path "
              f"'{pf_targets_file_path}'.")
    else:
        logger.warning("'pf_targets_file_path' is not set in the model "
              "config['data'] -> PFlog1pPF will use PER-FILE targets. This is "
              "correct only if the model was trained with per-file targets.")

    logger.info('STEP 2: TOKENIZING ANNDATA OBJECT...')
    # Tokenize adata
    if model_config['data']['tokenizer_type'] == 'cell_neigh':
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
        pf_targets_file_path=pf_targets_file_path,
        token_dictionary_file_path=token_dictionary_file_path,
        add_neigh_cell_ids=add_neigh_cell_ids,
        include_special_tokens=include_special_tokens)
    dataset_dict = tk._tokenize_adata(adata=adata)
    dataset = tk._create_dataset(
        dataset_dict=dataset_dict,
        use_generator=use_generator,
        cache_directory_path=cache_directory_path,
        keep_in_memory=keep_in_memory)

    columns = list(dataset.features.keys())
    columns.remove("cell_id")
    if "cell_ids" in columns:
        # Neighborhood cell-id strings (added when add_neigh_cell_ids=True) can't
        # be torch-formatted; keep them as a plain (output_all_columns) column.
        columns.remove("cell_ids")
    dataset.set_format(
        type="torch",
        columns=columns,
        output_all_columns=True)
    
    return dataset