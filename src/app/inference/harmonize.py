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


def harmonize_adata(adata: ad.AnnData,
                    gene_mapping_dict_file_path: str | None = '/lustre/scratch126/cellgen/lotfollahi/DATASETS/genes/homo_sapiens_gene_name_to_ensembl_id_dict.pkl',
                    gene_occurrence_count_file_path: str | None = '/lustre/scratch126/cellgen/lotfollahi/DATASETS/genes/homo_sapiens_gene_occurence_count_dict.pkl',
                    gene_occurrence_count_filter_value: int = 10,
                    ensembl_release: int = 111,
                    species: str = 'human',
                    min_genes_per_cell: int = 10,
                    min_cells_per_gene: int = 10,
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
    use_counts_from_layers = False
    while True:
        if issparse(adata.X):
            data = adata.X.data
        else:
            data = np.asarray(adata.X)
        all_integers = np.allclose(data, data.astype(int))

        if not all_integers:
            if 'counts' in adata.layers.keys():
                adata.X = adata.layers['counts']
                use_counts_from_layers = True
                continue
            else:
                raise ValueError(
                    "adata.X does not contain raw counts. "
                    "Found non-integer values in the count matrix."
                )
        else:
            break

    if use_counts_from_layers:
        print("Using counts from adata.layers['counts'] as adata.X"
              " did not contain raw counts (integer values).")
    else:
        print("✓ adata.X contains raw counts (integer values).")

    print('==================================================')
    print('STEP 2: ADDING ENSEMBL IDS...')
    print('==================================================')
    if not gene_mapping_dict_file_path:
        print(f'Adding ensembl IDs from release {ensembl_release}...')
        print(f'Make sure this ensembl release is aligned with pretraining.')
        print(f'Current ensembl release used for pretraining is 111.')
        # Extract ensembl IDs of protein coding and miRNA mouse genes
        ensembl = EnsemblRelease(release=ensembl_release, species=species)
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

    if gene_occurrence_count_file_path:
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
    n_cells_before = adata.n_obs
    print(f'Filtering cells with less than {min_genes_per_cell} genes.')
    sc.pp.filter_cells(
        adata,
        min_genes=min_genes_per_cell)
    n_cells_after = adata.n_obs
    print(f"Before cell filtering: {n_cells_before} cells.")
    print(f"After cell filtering: {n_cells_after} cells "
          f"(removed {n_cells_before - n_cells_after} cells).")

    # Filter genes with less than min_cells_per_gene cells
    n_genes_before = adata.n_vars
    print(f'Filtering genes expressed in less than {min_cells_per_gene} cells.')
    sc.pp.filter_genes(
        adata,
        min_cells=min_cells_per_gene)
    n_genes_after = adata.n_vars
    print(f"Before gene filtering: {n_genes_before} genes.")
    print(f"After gene filtering: {n_genes_after} genes "
          f"(removed {n_genes_before - n_genes_after} genes).")

    # Add dummy values as special values
    print('==================================================')
    print('STEP 4: ADDING SPECIAL VALUES...')
    print('==================================================')
    if 'cell_id' not in adata.obs.keys():
        adata.obs['cell_id'] = adata.obs_names
    if 'dataset_id' not in adata.uns.keys():
        adata.uns['dataset_id'] = 14  # just dummy value to prevent errors
    if 'batch' not in adata.uns.keys():
        adata.uns['batch'] = 'batch0' # just dummy value to prevent errors
    if 'assay' not in adata.uns.keys():
        adata.uns['assay'] = 'xenium' # just dummy value to prevent errors
    if 'species' not in adata.uns.keys():
        adata.uns['species'] = 'homo_sapiens' # just dummy value to prevent errors
    if 'tissue' not in adata.uns.keys():
        adata.uns['tissue'] = 'lung' # just dummy value to prevent errors

    return adata