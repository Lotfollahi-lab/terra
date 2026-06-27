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


def _build_perturb_index(df: pd.DataFrame) -> dict[str, list[dict]]:
    """
    Turn perturb_df into an index:
        {perturbed_cell_id: [row_as_dict, …]}
    (Using itertuples avoids Python object creation for every column.)
    """
    index = defaultdict(list)
    for row in df.itertuples(index=False):
        index[row.perturbed_cell_id].append(row._asdict())
    return index

def _perturb_batch_with_idx(
    batch: dict,
    index: dict[str, list[dict]],
    seq_len_cell: int = 256,
    ) -> dict:
    """
    Modify the tensors in-place (cheap) and return the same dict.

    Parameters
    -----------
    batch:
        The dictionary mapping column -> list-of-values returned by huggingface
        when batched=True.
    """
    B = len(batch["cell_ids"]) # batch size
    for b in range(B):
        cell_ids = list(dict.fromkeys(batch["cell_ids"][b]))
        # Fast reject: does this batch touch any perturbed cell at all?
        if not any(cid in index for cid in cell_ids):
            continue

        # Handle index-cell (first) and neighbourhood (rest) separately
        for cid in cell_ids:
            for row in index.get(cid, []):
                is_index_cell = (cid == cell_ids[0])
                if row["perturbation_target"] == "cell" and not is_index_cell:
                    continue
                if row["perturbation_target"] == "neighborhood" and is_index_cell:
                    continue

                gene_tokens = batch["gene_tokens"][b]
                gene_expr   = batch["gene_expr"][b]

                # --- choose which token positions to perturb -------------
                if row["perturbed_gene_token"] == "all":
                    if is_index_cell:
                        idx = slice(0, seq_len_cell)
                    else:
                        idx = slice(seq_len_cell, None)
                else:
                    token_id = row["perturbed_gene_token"]
                    token_slice = (
                        gene_tokens[:seq_len_cell]
                        if is_index_cell else
                        gene_tokens[seq_len_cell:]
                    )
                    rel_idx = torch.nonzero(token_slice == token_id, as_tuple=True)[0]
                    offset  = 0 if is_index_cell else seq_len_cell
                    idx     = rel_idx + offset
                # ---------------------------------------------------------

                if row["perturbation_type"] == "knockout":
                    gene_expr[idx] = 0.0
                elif row["perturbation_type"] == "foldchange":
                    gene_expr[idx] *= row["foldchange"]
                else:
                    raise ValueError(f"Bad perturbation_type: {row['perturbation_type']}")

    return batch

def _perturb_batch_with_df(
    batch: dict,
    df: pd.DataFrame,
    seq_len_cell: int = 256,
    n_segments: int = 11,
    pad_gene_tokens: bool = True,
    adjust_positions: bool = False,
    ) -> dict:
    """
    Modify batch of token sequences in place based on config defined in
    perturbation dataframe.

    Parameters
    -----------
    batch:
        Batch of token sequences, a dictionary mapping field -> tensor of
        tokens, returned by huggingface when batched is `True`.
    df:
        Dataframe containing the perturbation config.
    seq_len_cell:
        Number of cell gene tokens (excluding neighborhood gene tokens).

    Returns
    -----------
    batch:
        Batch of perturbed token sequences, modified in place.
    """
    for idx, row in df.iterrows():
        # Validate perturbation dataframe
        if row["perturbation_target"] not in ['cell', 'neighborhood']:
            raise ValueError(
                f"Invalid perturbation_target: {row['perturbation_target']}.")
        if row["perturbation_type"] not in ['knockout', 'foldchange']:
            raise ValueError(
                f"Bad perturbation_type: {row['perturbation_type']}")

        # Get indices of tokens to be perturbed
        cell_perturbation = row["perturbation_target"] == 'cell'
        if row["perturbed_gene_token"] == "all":
            # Perturb every gene position in the cell (or neighborhood) segment,
            # for every cell in the batch. The advanced-indexing path below is
            # only valid for a specific gene token, so apply "all" directly.
            col = (slice(0, seq_len_cell) if cell_perturbation
                   else slice(seq_len_cell, None))
            if row["perturbation_type"] == "knockout":
                batch["gene_expr"][:, col] = 0.0
                if pad_gene_tokens:
                    batch["gene_tokens"][:, col] = 0
            else:  # foldchange
                batch["gene_expr"][:, col] *= row["foldchange"]
            batch[f'pert_flag_idx_{idx}'] = torch.ones(
                batch["gene_tokens"].size(0), dtype=torch.bool)
            continue

        token_id = row["perturbed_gene_token"]
        token_slice = (
            batch["gene_tokens"][:, :seq_len_cell] if cell_perturbation
            else batch["gene_tokens"][:, seq_len_cell:])
        cell_pert_idx, rel_gene_pert_idx = torch.nonzero(
            token_slice == token_id, as_tuple=True)
        offset = 0 if cell_perturbation else seq_len_cell
        abs_gene_pert_idx = rel_gene_pert_idx + offset

        # Perturb tokens and track perturbation flags
        batch[f'pert_flag_idx_{idx}'] = torch.zeros(
            batch["gene_tokens"].size(0),
            dtype=torch.bool)
        batch[f'pert_flag_idx_{idx}'][cell_pert_idx] = True
        if row["perturbation_type"] == "knockout":
            batch["gene_expr"][cell_pert_idx, abs_gene_pert_idx] = 0.0
            if pad_gene_tokens:
                batch["gene_tokens"][cell_pert_idx, abs_gene_pert_idx] = 0
        elif row["perturbation_type"] == "foldchange":
            batch["gene_expr"][
                cell_pert_idx, abs_gene_pert_idx] *= row["foldchange"]

        if adjust_positions:
            gt = batch["gene_tokens"] # (B, n_segments*seq_len_cell)
            ge = batch["gene_expr"] # (B, n_segments*seq_len_cell)

            B = gt.shape[0]
            gt = gt.reshape(B, n_segments, seq_len_cell)
            ge = ge.reshape(B, n_segments, seq_len_cell)

            # mask: True where token==0; sort so False first, True last (stable keeps order)
            mask = (gt == 0)

            # indices shape: (B, n_segments, seq_len_cell)
            idx = torch.argsort(mask.to(torch.int64), dim=-1, stable=True)

            # reorder both tensors with same indices
            gt_sorted = torch.gather(gt, dim=-1, index=idx)
            ge_sorted = torch.gather(ge, dim=-1, index=idx)

            # flatten back to (B, n_segments*seq_len_cell)
            batch["gene_tokens"] = gt_sorted.reshape(B, n_segments * seq_len_cell)
            batch["gene_expr"] = ge_sorted.reshape(B, n_segments * seq_len_cell)

        #if len(cell_pert_idx) == 0:
        #else:

    return batch

def perturb_dataset(dataset: Dataset,
                    perturb_df: pd.DataFrame,
                    model_folder_path: str,
                    seq_len_cell: int = 256,
                    n_segments: int = 11,
                    nproc: int = 4,
                    batch_size: int = 1000,
                    keep_in_memory: bool = False,
                    return_only_perturbed_cells: bool = False,
                    pad_gene_tokens: bool = True,
                    adjust_positions: bool = False,
                    ) -> Dataset:
    """
    Perturb a huggingface dataset.
    """
    # Load token dictionary
    with open(Path(model_folder_path) / "token_dictionary.pkl", "rb") as f:
        token_dict = pickle.load(f)

    # Convert ensembl IDs to token IDs, keeping "all"
    perturb_df = perturb_df.copy()
    perturb_df["perturbed_gene_token"] = perturb_df["perturbed_ensembl_id"].where(
        perturb_df["perturbed_ensembl_id"] == "all",
        perturb_df["perturbed_ensembl_id"].map(token_dict),
    )
    logger.info(f"Applying perturbations using dataframe:\n{perturb_df}")

    # If all perturbations are on all cells, skip indexing
    perturbed_cell_ids = perturb_df["perturbed_cell_id"].unique().tolist()
    if len(perturbed_cell_ids) == 1 and perturbed_cell_ids[0] == "all":

        # Use partial so the dataset mapper sees only one argument as expected
        perturb_fn = partial(
            _perturb_batch_with_df,
            df=perturb_df,
            seq_len_cell=seq_len_cell,
            n_segments=n_segments,
            pad_gene_tokens=pad_gene_tokens,
            adjust_positions=adjust_positions
        )
    
    else:
        # Build an index for fast cell lookup
        perturb_index = _build_perturb_index(perturb_df)

        # Use partial so the dataset mapper sees only one argument as expected
        perturb_fn = partial(
            _perturb_batch_with_idx,
            index=perturb_index,
            seq_len_cell=seq_len_cell,
        )

    # Map in batch mode
    dataset = dataset.map(
        perturb_fn,
        batched=True,
        batch_size=batch_size,
        num_proc=nproc,
        keep_in_memory=keep_in_memory,
        load_from_cache_file=False)

    # Optionally, return only perturbed cells. Read the per-row perturbation
    # flags and select the matching indices -- far cheaper than filter(), which
    # scans and rewrites every (large) token row to test one field.
    if return_only_perturbed_cells:
        pert_cols = [
            c for c in dataset.column_names if c.startswith("pert_flag_idx")]
        flags = np.column_stack(
            [np.asarray(dataset[c], dtype=bool) for c in pert_cols])
        keep_idx = np.nonzero(flags.any(axis=1))[0]
        dataset = dataset.select(keep_idx)

    return dataset