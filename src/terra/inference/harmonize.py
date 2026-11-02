import copy
import logging
import os
import pickle
import re
import warnings
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

# Ensembl gene ID with an optional version suffix, e.g. ENSG00000141510.17
_ENSEMBL_GENE_ID = re.compile(r'(ENS[A-Z]*G\d+)(\.\d+)?')

# Fraction of genes below which harmonization warns that almost no gene maps
_MIN_MAPPED_GENE_FRACTION = 0.1

# Number of duplicated genes named in the warning about them
_MAX_LISTED_GENES = 10


def _contains_integers(x) -> bool:
    """Return `True` if all (stored) values of a count matrix are integers."""
    data = x.data if issparse(x) else np.asarray(x)
    return np.allclose(data, data.astype(int))


def _map_ensembl_ids(gene_names: list[str],
                     ensembl_ids: set[str],
                     ) -> dict:
    """
    Map gene names that are Ensembl IDs (with or without version suffix) to
    the Ensembl IDs in `ensembl_ids`.
    """
    ensembl_id_map_dict = {}
    for gene_name in gene_names:
        match = _ENSEMBL_GENE_ID.fullmatch(gene_name)
        if match and match.group(1) in ensembl_ids:
            ensembl_id_map_dict[gene_name] = match.group(1)
    return ensembl_id_map_dict


def harmonize_adata(adata: ad.AnnData,
                    gene_mapping_dict_file_path: str | None = None,
                    gene_occurrence_count_file_path: str | None = None,
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
    gene_mapping_dict_file_path:
        Path to a pickled gene-name-to-ensembl-ID mapping dictionary. If `None`,
        ensembl IDs are looked up from the specified Ensembl release instead.
        `adata.var_names` can be gene names (symbols) or Ensembl gene IDs
        (optionally with version suffix); Ensembl IDs are used directly if
        they match more genes of the mapping than the gene names do. A gene
        in more than one column (e.g. duplicated `var_names`) is kept in
        every column, with a warning.
    gene_occurrence_count_file_path:
        Path to a pickled dictionary of per-gene pretraining occurrence counts. If
        provided, genes with too few occurrences are filtered out; if `None`, this
        filtering step is skipped.
    gene_occurrence_count_filter_value:
        Minimum number of pretraining occurrences a gene must have to be kept.
    ensembl_release:
        Ensembl release used to retrieve ensembl IDs.
    species:
        Species passed to the Ensembl release lookup.
    min_genes_per_cell:
        Minimum amount of genes per cell for a cell not to be filtered.
    min_cells_per_gene:
        Minimum number of cells a gene must be expressed in for the gene not to be
        filtered.

    Returns
    -----------
    adata : anndata.AnnData
        A harmonized AnnData object (a new object; `adata` is not modified).
        If `adata.X` does not contain raw counts, its `X` holds the counts of
        `adata.layers['counts']`.
    """
    logger.info('STEP 1: DATA VALIDATION...')
    logger.info('Checking that adata.X contains raw counts...')
    use_counts_from_layers = not _contains_integers(adata.X)
    if use_counts_from_layers:
        if 'counts' not in adata.layers.keys():
            raise ValueError(
                "adata.X does not contain raw counts. "
                "Found non-integer values in the count matrix."
            )
        if not _contains_integers(adata.layers['counts']):
            raise ValueError(
                "Neither adata.X nor adata.layers['counts'] contains raw "
                "counts. Found non-integer values in both count matrices."
            )

    if use_counts_from_layers:
        logger.warning("Using counts from adata.layers['counts'] as adata.X"
              " did not contain raw counts (integer values).")
    else:
        logger.info("✓ adata.X contains raw counts (integer values).")

    logger.info('STEP 2: ADDING ENSEMBL IDS...')
    if not gene_mapping_dict_file_path:
        logger.info(f'Adding ensembl IDs from release {ensembl_release}...')
        logger.info(f'Make sure this ensembl release is aligned with pretraining.')
        logger.info(f'Current ensembl release used for pretraining is 111.')
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
        logger.info(f'Adding ensembl IDs from gene_mapping_dict with file path `{gene_mapping_dict_file_path}`...')
        with open(gene_mapping_dict_file_path, 'rb') as f:
            gene_ensembl_map_dict = pickle.load(f)

    adata_gene_names = [gene_name for gene_name in adata.var_names.tolist()]

    # adata.var_names may be Ensembl IDs instead of gene names (e.g. 10x
    # gene_ids); these are matched against the Ensembl IDs of the mapping
    n_matching_gene_names = sum(
        gene_name in gene_ensembl_map_dict for gene_name in adata_gene_names)
    ensembl_id_map_dict = _map_ensembl_ids(
        adata_gene_names, set(gene_ensembl_map_dict.values()))
    if len(ensembl_id_map_dict) > n_matching_gene_names:
        logger.info('adata.var_names are Ensembl IDs; using them directly.')
        gene_ensembl_map_dict = ensembl_id_map_dict

    harmonized_gene_names = []
    matching_ensembl_ids = []
    for gene_name in adata_gene_names:
        if gene_name in gene_ensembl_map_dict.keys():
            harmonized_gene_names.append(gene_name)
            matching_ensembl_ids.append(gene_ensembl_map_dict[gene_name])
    logger.info(f'Number of genes with matching ensembl IDs: {len(harmonized_gene_names)}.')
    logger.warning(f'Number of genes skipped due to non-matching ensembl IDs: {len(adata_gene_names) - len(harmonized_gene_names)}.')
    if len(adata_gene_names) - len(harmonized_gene_names) > 0:
        logger.warning(f'Genes excluded due to non-matching ensembl IDs: {set(adata_gene_names) - set(harmonized_gene_names)}.')
    if not harmonized_gene_names:
        raise ValueError(
            f'None of the {len(adata_gene_names)} genes in adata.var_names '
            f'(e.g. {adata_gene_names[:3]}) maps to an Ensembl ID of the '
            'model. adata.var_names must be gene names (symbols) or Ensembl '
            "gene IDs of the model's species.")
    if (len(harmonized_gene_names)
            < _MIN_MAPPED_GENE_FRACTION * len(adata_gene_names)):
        logger.warning(
            f'Only {len(harmonized_gene_names)} of {len(adata_gene_names)} '
            'genes in adata.var_names map to an Ensembl ID of the model; all '
            'other genes are dropped. Check that adata.var_names are gene '
            "names (symbols) or Ensembl gene IDs of the model's species.")

    # A gene in several columns (e.g. duplicated var_names) occurs several
    # times in the token sequence of a cell. The columns are kept, which
    # reproduces existing results, with a warning (shown without logging
    # configuration)
    ensembl_id_gene_names = defaultdict(list)
    for gene_name, ensembl_id in zip(harmonized_gene_names,
                                     matching_ensembl_ids):
        ensembl_id_gene_names[ensembl_id].append(gene_name)
    duplicated = {ensembl_id: gene_names for ensembl_id, gene_names
                  in ensembl_id_gene_names.items() if len(gene_names) > 1}
    if duplicated:
        examples = '; '.join(
            f"{ensembl_id}: {', '.join(gene_names)}" for ensembl_id, gene_names
            in list(duplicated.items())[:_MAX_LISTED_GENES])
        if len(duplicated) > _MAX_LISTED_GENES:
            examples += '; ...'
        warnings.warn(
            f'{len(duplicated)} genes occur in more than one column of adata '
            f'({examples}). Every column is kept, so these genes occur more '
            'than once in the token sequence of a cell. To keep each gene '
            'once, make adata.var_names unique (e.g. with '
            'adata.var_names_make_unique()) or sum the counts of the '
            'duplicated columns before harmonization.',
            UserWarning, stacklevel=2)

    # Subset into a new object; the caller's AnnData is never modified
    adata = adata[:, adata.var.index.isin(harmonized_gene_names)].copy()
    if use_counts_from_layers:
        adata.X = adata.layers['counts'].copy()
    adata.var = pd.DataFrame(
        index=pd.Index(harmonized_gene_names, name='gene_name'),
        data={'ensembl_id': matching_ensembl_ids})

    if gene_occurrence_count_file_path:
        logger.info(f'Filtering genes that have not occurred enough during pretraining...')
        with open(gene_occurrence_count_file_path, 'rb') as f:
            gene_occurrence_count_dict = pickle.load(f)

        all_ensembl_ids = adata.var['ensembl_id'].tolist()
        # A gene missing from the count dictionary did not occur in
        # pretraining
        n_missing = sum(ensembl_id not in gene_occurrence_count_dict
                        for ensembl_id in all_ensembl_ids)
        if n_missing > 0:
            logger.info(f'Number of genes without pretraining occurrence count (treated as 0): {n_missing}.')
        keep_ensembl_ids = [
            ensembl_id for ensembl_id in all_ensembl_ids if gene_occurrence_count_dict.get(
                    ensembl_id, 0) > gene_occurrence_count_filter_value]

        logger.warning(f'Number of genes skipped due to not enough pretraining occurrences: {len(all_ensembl_ids) - len(keep_ensembl_ids)}.')
        if len(all_ensembl_ids) - len(keep_ensembl_ids) > 0:
            logger.warning(f'Genes excluded due to not enough pretraining occurrences: {set(all_ensembl_ids) - set(keep_ensembl_ids)}.')
        adata = adata[:, adata.var['ensembl_id'].isin(keep_ensembl_ids)].copy()

    logger.info('STEP 3: BASIC QUALITY CONTROL...')
    # Filter cells with less than min_genes_per_cell genes
    n_cells_before = adata.n_obs
    logger.info(f'Filtering cells with less than {min_genes_per_cell} genes.')
    sc.pp.filter_cells(
        adata,
        min_genes=min_genes_per_cell)
    n_cells_after = adata.n_obs
    logger.info(f"Before cell filtering: {n_cells_before} cells.")
    logger.info(f"After cell filtering: {n_cells_after} cells "
          f"(removed {n_cells_before - n_cells_after} cells).")

    # Filter genes with less than min_cells_per_gene cells
    n_genes_before = adata.n_vars
    logger.info(f'Filtering genes expressed in less than {min_cells_per_gene} cells.')
    sc.pp.filter_genes(
        adata,
        min_cells=min_cells_per_gene)
    n_genes_after = adata.n_vars
    logger.info(f"Before gene filtering: {n_genes_before} genes.")
    logger.info(f"After gene filtering: {n_genes_after} genes "
          f"(removed {n_genes_before - n_genes_after} genes).")

    if 'cell_id' not in adata.obs.keys():
        adata.obs['cell_id'] = adata.obs_names

    return adata