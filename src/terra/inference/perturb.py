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
    for order, row in enumerate(df.itertuples(index=False)):
        request = row._asdict()
        request["_perturbation_order"] = order
        index[row.perturbed_cell_id].append(request)
    return index


def _fold_change(values: torch.Tensor,
                 foldchange: float,
                 log1p_values: bool,
                 ) -> torch.Tensor:
    """
    Expression values changed by `foldchange`.

    Log1p-normalized values (``shifted_log``, i.e. log1p(counts)) are changed
    in count space, ``log1p(expm1(values) * foldchange)``, so that a fold
    change of 2 doubles the counts; other values are multiplied directly.
    """
    if log1p_values:
        return torch.log1p(torch.expm1(values) * foldchange)
    return values * foldchange

def _perturb_batch_with_idx(
    batch: dict,
    index: dict[str, list[dict]],
    seq_len_cell: int = 256,
    pad_gene_tokens: bool = True,
    log1p_values: bool = False,
    adjust_positions: bool = False,
    n_segments: int = 11,
) -> dict:
    """
    Modify the tensors in-place (cheap) and return the same dict.

    Cells are matched by their own ``cell_id`` for ``cell``-target
    perturbations. ``neighborhood``-target perturbations of a specific cell
    edit, in the sequences of the cells that have it as a neighbor, the
    positions of its own tokens; they need the per-token neighbor IDs (the
    ``cell_ids`` column added by ``add_neigh_cell_ids``) and are skipped if
    that column is absent.

    Parameters
    -----------
    batch:
        The dictionary mapping column -> list-of-values returned by Hugging Face
        when batched=True.
    index:
        Perturbations per perturbed cell ID (``_build_perturb_index``).
    seq_len_cell:
        Number of cell gene tokens (excluding neighborhood gene tokens).
    pad_gene_tokens:
        If `True`, also zero out the gene tokens of knocked-out genes.
    log1p_values:
        If `True`, the expression values are log1p-normalized counts and
        fold changes apply to the counts (``_fold_change``).
    adjust_positions:
        For mixed wildcard/specific requests, pack padding to the end of each
        segment once all edits have been applied. Specific-only behavior is
        unchanged.
    n_segments:
        Number of fixed-length segments when packing mixed requests.
    """
    # Mapped without the torch format; convert the edited columns to tensors so
    # the in-place per-cell edits below work (no-op if already tensors).
    batch["gene_tokens"] = torch.as_tensor(np.asarray(batch["gene_tokens"]))
    batch["gene_expr"] = torch.as_tensor(np.asarray(batch["gene_expr"]))
    own_ids = batch["cell_id"]            # each cell's own id (always present)
    neigh_ids = batch.get("cell_ids")     # neighbor ids (add_neigh_cell_ids only)
    B = len(own_ids)
    cell_positions = torch.arange(seq_len_cell)
    wildcard_rows = index.get("all", [])
    neighborhood_positions = torch.arange(seq_len_cell, batch["gene_tokens"].shape[1])

    def _apply(b, row, positions):
        gene_tokens = batch["gene_tokens"][b]
        gene_expr = batch["gene_expr"][b]
        if row["perturbed_gene_token"] == "all":
            idx = positions
        else:
            idx = positions[
                gene_tokens[positions] == row["perturbed_gene_token"]]
        if row["perturbation_type"] == "knockout":
            gene_expr[idx] = 0.0
            if pad_gene_tokens:
                gene_tokens[idx] = 0
        elif row["perturbation_type"] == "foldchange":
            gene_expr[idx] = _fold_change(
                gene_expr[idx], row["foldchange"], log1p_values)
        else:
            raise ValueError(f"Bad perturbation_type: {row['perturbation_type']}")

    for b in range(B):
        own = own_ids[b]
        operations = []
        # cell-target: edit cell b's own tokens when b is in the perturb list
        for row in index.get(own, []):
            if row["perturbation_target"] == "cell":
                operations.append((row, cell_positions))
        # neighborhood-target: edit the tokens of each listed neighbor of b
        if neigh_ids is not None:
            neighbors = np.asarray(neigh_ids[b][seq_len_cell:])
            for nid in dict.fromkeys(neighbors.tolist()):
                if nid == own:
                    continue
                rows = [row for row in index.get(nid, [])
                        if row["perturbation_target"] == "neighborhood"]
                if not rows:
                    continue
                positions = torch.as_tensor(
                    np.nonzero(neighbors == nid)[0] + seq_len_cell)
                for row in rows:
                    operations.append((row, positions))
        if wildcard_rows:
            for row in wildcard_rows:
                target = row["perturbation_target"]
                if target not in ("cell", "neighborhood"):
                    raise ValueError(f"Invalid perturbation_target: {target}.")
                positions = cell_positions if target == "cell" else neighborhood_positions
                operations.append((row, positions))
            # Merge scopes by original row ordinal, including repeated rows.
            operations.sort(key=lambda operation: operation[0]["_perturbation_order"])
        for row, positions in operations:
            _apply(b, row, positions)

    if wildcard_rows and adjust_positions:
        # Move token/value pairs only within fixed cell/neighbor segments.
        # Provenance IDs are constant within these segments and stay unchanged.
        tokens = batch["gene_tokens"].reshape(B, n_segments, seq_len_cell)
        values = batch["gene_expr"].reshape(B, n_segments, seq_len_cell)
        order = torch.argsort((tokens == 0).to(torch.int64), dim=-1, stable=True)
        batch["gene_tokens"] = torch.gather(tokens, -1, order).reshape(B, -1)
        batch["gene_expr"] = torch.gather(values, -1, order).reshape(B, -1)
    # Return the edited columns as numpy arrays. datasets re-encodes numpy via
    # fast buffer copies; torch tensors are ~2x slower and python lists are
    # pathologically slow (can appear to hang) for these wide token columns.
    batch["gene_tokens"] = batch["gene_tokens"].numpy()
    batch["gene_expr"] = batch["gene_expr"].numpy()
    # Drop the large, unmodified neighbor-ID column from the output (perturb_dataset
    # re-attaches the original); absent unless this batch needed it as input.
    batch.pop("cell_ids", None)
    return batch

def _perturb_batch_with_df(
    batch: dict,
    df: pd.DataFrame,
    seq_len_cell: int = 256,
    n_segments: int = 11,
    pad_gene_tokens: bool = True,
    adjust_positions: bool = False,
    log1p_values: bool = False,
    ) -> dict:
    """
    Modify batch of token sequences in place based on config defined in
    perturbation dataframe.

    Only the gene tokens/expression are edited; no extra columns are added, so
    the map output keeps the input schema (adding columns would force
    ``datasets`` to re-encode every column, including any large nested
    ``cell_ids`` column, and can stall). ``return_only_perturbed_cells`` is
    handled separately via ``_affected_cell_indices``.

    Parameters
    -----------
    batch:
        Batch of token sequences, a dictionary mapping field -> tensor of
        tokens, returned by Hugging Face when batched is `True`.
    df:
        Dataframe containing the perturbation config.
    seq_len_cell:
        Number of cell gene tokens (excluding neighborhood gene tokens).
    log1p_values:
        If `True`, the expression values are log1p-normalized counts and
        fold changes apply to the counts (``_fold_change``).

    Returns
    -----------
    batch:
        Batch of perturbed token sequences, modified in place.
    """
    # The dataset is mapped without its torch format (formatting the large
    # nested token columns inside .map can stall), so convert the two columns we
    # edit to tensors here -- datasets writes the modified tensors back out.
    # ``as_tensor`` is a no-op if they are already tensors.
    batch["gene_tokens"] = torch.as_tensor(np.asarray(batch["gene_tokens"]))
    batch["gene_expr"] = torch.as_tensor(np.asarray(batch["gene_expr"]))
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
                batch["gene_expr"][:, col] = _fold_change(
                    batch["gene_expr"][:, col], row["foldchange"],
                    log1p_values)
            continue

        token_id = row["perturbed_gene_token"]
        token_slice = (
            batch["gene_tokens"][:, :seq_len_cell] if cell_perturbation
            else batch["gene_tokens"][:, seq_len_cell:])
        cell_pert_idx, rel_gene_pert_idx = torch.nonzero(
            token_slice == token_id, as_tuple=True)
        offset = 0 if cell_perturbation else seq_len_cell
        abs_gene_pert_idx = rel_gene_pert_idx + offset

        if row["perturbation_type"] == "knockout":
            batch["gene_expr"][cell_pert_idx, abs_gene_pert_idx] = 0.0
            if pad_gene_tokens:
                batch["gene_tokens"][cell_pert_idx, abs_gene_pert_idx] = 0
        elif row["perturbation_type"] == "foldchange":
            batch["gene_expr"][cell_pert_idx, abs_gene_pert_idx] = (
                _fold_change(
                    batch["gene_expr"][cell_pert_idx, abs_gene_pert_idx],
                    row["foldchange"], log1p_values))

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

    # Return the edited columns as numpy arrays. datasets re-encodes numpy via
    # fast buffer copies; torch tensors are ~2x slower and python lists are
    # pathologically slow (can appear to hang) for these wide token columns.
    batch["gene_tokens"] = batch["gene_tokens"].numpy()
    batch["gene_expr"] = batch["gene_expr"].numpy()
    # Drop the large, unmodified neighbor-ID column from the output (perturb_dataset
    # re-attaches the original); absent unless this batch needed it as input.
    batch.pop("cell_ids", None)
    return batch


def _affected_cell_indices(dataset: Dataset,
                           perturb_df: pd.DataFrame,
                           seq_len_cell: int,
                           return_masks: bool = False):
    """Row indices of cells whose tokens the perturbations actually edit.

    Computed analytically from ``perturb_df`` (plus gene presence) instead of
    emitting per-row flag columns during the map -- adding columns changes the
    map output schema and forces ``datasets`` to re-encode every column,
    including any large nested ``cell_ids`` column. ``dataset`` must be the
    *unperturbed* dataset (knockout zeroes the gene tokens, so gene presence has
    to be read before perturbing).

    A cell is affected by a row when it is targeted by that row -- itself for a
    ``cell`` target, or having a listed cell as a neighbor for a
    ``neighborhood`` target (``"all"`` targets every cell) -- and, for a
    specific gene, that gene token is present in the relevant segment (the
    cell's own gene tokens for a ``cell`` target, the neighborhood tokens for a
    ``neighborhood`` target, and only those of the listed neighbor for a
    specific cell).
    """
    n = len(dataset)
    affected = np.zeros(n, dtype=bool)
    masks = []  # per-row masks, only used when return_masks=True
    gene_tokens = cell_id_col = neigh_lists = None  # read lazily, once
    for _, row in perturb_df.iterrows():
        cell_target = row["perturbation_target"] == "cell"
        tok = row["perturbed_gene_token"]

        neighbor_positions = None
        if row["perturbed_cell_id"] == "all":
            targeted = np.ones(n, dtype=bool)
        elif cell_target:
            if cell_id_col is None:
                cell_id_col = np.asarray([str(c) for c in dataset["cell_id"]])
            targeted = cell_id_col == str(row["perturbed_cell_id"])
        else:
            # neighborhood target on a specific cell: cells having it as
            # neighbor, at the positions of its tokens
            if neigh_lists is None:
                neigh_lists = dataset["cell_ids"]
            pid = str(row["perturbed_cell_id"])
            neighbor_positions = [
                np.asarray([str(x) for x in ids[seq_len_cell:]]) == pid
                for ids in neigh_lists]
            targeted = np.fromiter(
                (positions.any() for positions in neighbor_positions),
                dtype=bool, count=n)

        if tok != "all":
            if gene_tokens is None:
                gene_tokens = np.asarray(dataset["gene_tokens"])
            seg = (gene_tokens[:, :seq_len_cell] if cell_target
                   else gene_tokens[:, seq_len_cell:])
            present = seg == int(tok)
            if neighbor_positions is None:
                targeted = targeted & present.any(axis=1)
            else:
                targeted = targeted & np.fromiter(
                    (present[i, :len(positions)][positions].any()
                     for i, positions in enumerate(neighbor_positions)),
                    dtype=bool, count=n)

        affected |= targeted
        if return_masks:
            masks.append(targeted)
    idx = np.nonzero(affected)[0]
    if return_masks:
        mask_matrix = (np.column_stack(masks) if masks
                       else np.zeros((n, 0), dtype=bool))
        return idx, mask_matrix
    return idx


def perturb_dataset(dataset: Dataset,
                    perturb_df: pd.DataFrame,
                    model_folder_path: str,
                    seq_len_cell: int = 256,
                    n_segments: int = 11,
                    nproc: int = 1,
                    batch_size: int = 1000,
                    keep_in_memory: bool = False,
                    return_only_perturbed_cells: bool = False,
                    pad_gene_tokens: bool = True,
                    adjust_positions: bool = False,
                    return_perturbation_flags: bool = False,
                    ) -> Dataset:
    """
    Perturb a tokenized Hugging Face dataset according to a perturbation dataframe.

    The perturbations described in `perturb_df` (gene knockouts or fold changes,
    applied to a cell's own gene tokens or to its neighborhood) are applied to
    the gene tokens/expression of the dataset. Perturbations on all cells are
    applied batch-wise; perturbations targeting specific cells are matched by
    `cell_id` (and, for neighborhood targets, by the stored neighbor IDs).
    Mixed wildcard and specific-cell requests apply in DataFrame row order;
    repeated rows are applied repeatedly, without deduplication.

    Parameters
    -----------
    dataset:
        Tokenized Hugging Face dataset to perturb. Perturbing the neighborhood of
        specific cells requires the dataset to have been tokenized with
        `add_neigh_cell_ids=True` (so neighbor IDs are stored).
    perturb_df:
        Dataframe describing the perturbations. Expected columns are
        `perturbed_cell_id` (a cell ID or `"all"`), `perturbed_ensembl_id` (an
        ensembl ID or `"all"`), `perturbation_target` (`"cell"` or
        `"neighborhood"`) and `perturbation_type` (`"knockout"` or
        `"foldchange"`), plus a `foldchange` value for fold-change rows. A
        fold change multiplies the counts: for models whose expression values
        are log1p-normalized (`count_count_norm_method` 'shifted_log', e.g.
        TERRA-112M and TERRA-96M) a value v becomes
        log1p(expm1(v) * foldchange); other values are multiplied directly. A
        `"neighborhood"` target on a specific cell edits that cell's tokens in
        the sequences of the cells that have it as a neighbor.
    model_folder_path:
        Path to the folder containing the model config and token dictionary;
        used to map ensembl IDs to token IDs.
    seq_len_cell:
        Number of cell gene tokens (excluding neighborhood gene tokens).
    n_segments:
        Number of segments (the index cell plus its neighbors); used when
        re-packing positions after perturbation.
    nproc:
        Number of processes used to map the perturbation over the dataset.
        Defaults to 1 (single process). Values >1 use multiprocessing, which can
        deadlock once torch/CUDA has been initialized in the process (the worker
        fork inherits a broken CUDA state) and adds dataset-serialization
        overhead; the perturbation itself is fast single-process.
    batch_size:
        Number of rows per batch passed to the mapping function.
    keep_in_memory:
        Whether to keep the perturbed dataset in memory during mapping.
    return_only_perturbed_cells:
        If `True`, return the union of cells eligible for the requests before
        any edits. Eligibility does not require a numerical change, for
        example after an earlier knockout of the same gene.
    pad_gene_tokens:
        If `True`, also zero out the gene tokens (not just the expression) of
        knocked-out genes, for all cells or specific cells.
    adjust_positions:
        If `True`, re-pack each segment so padded (zeroed) gene positions are
        moved to the end after perturbation.
        Mixed wildcard/specific requests pack once after all edits, preserving
        token/value pairs and fixed-segment neighbor IDs. The existing
        specific-only indexed path does not re-pack positions.
    return_perturbation_flags:
        If `True`, attach one boolean column per `perturb_df` row,
        `pert_flag_idx_{i}`, which is `True` for cells eligible for row `i`
        before any edits (its target gene is present in the targeted scope).
        Repeated rows can stay eligible after an earlier knockout; filtering
        uses the union of these pre-edit masks. The
        columns are attached after mapping (a zero-copy column concat), not
        inside `.map`, so the map output schema is unchanged. Defaults to
        `False`, in which case no extra columns are added.

    Returns
    -----------
    dataset : datasets.Dataset
        The perturbed Hugging Face `Dataset`, filtered to the affected cells if
        `return_only_perturbed_cells` is `True`.
    """
    # Load token dictionary
    with open(Path(model_folder_path) / "token_dictionary.pkl", "rb") as f:
        token_dict = pickle.load(f)

    # Expression values of shifted-log models are log1p(counts): fold changes
    # apply to the counts
    with open(Path(model_folder_path) / "model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    log1p_values = (
        model_config["data"].get("count_count_norm_method") == "shifted_log")

    # Convert ensembl IDs to token IDs, keeping "all"
    perturb_df = perturb_df.copy()
    perturb_df["perturbed_gene_token"] = perturb_df["perturbed_ensembl_id"].where(
        perturb_df["perturbed_ensembl_id"] == "all",
        perturb_df["perturbed_ensembl_id"].map(token_dict),
    )
    logger.info(f"Applying perturbations using dataframe:\n{perturb_df}")

    # If all perturbations are on all cells, skip indexing
    perturbed_cell_ids = perturb_df["perturbed_cell_id"].unique().tolist()
    all_cells = len(perturbed_cell_ids) == 1 and perturbed_cell_ids[0] == "all"
    needs_cell_ids_input = (
        (perturb_df["perturbation_target"] == "neighborhood") & (perturb_df["perturbed_cell_id"] != "all")
    ).any()
    if all_cells:

        # Use partial so the dataset mapper sees only one argument as expected
        perturb_fn = partial(
            _perturb_batch_with_df,
            df=perturb_df,
            seq_len_cell=seq_len_cell,
            n_segments=n_segments,
            pad_gene_tokens=pad_gene_tokens,
            adjust_positions=adjust_positions,
            log1p_values=log1p_values,
        )
    
    else:
        # Perturbing specific cells: cell-target rows are matched on the cell's
        # own `cell_id`; neighborhood-target rows need the neighbor IDs, which
        # only exist when the dataset was tokenized with add_neigh_cell_ids=True.
        if needs_cell_ids_input and "cell_ids" not in dataset.column_names:
            raise ValueError(
                "Perturbing the neighborhood of specific cells requires the "
                "dataset to be tokenized with add_neigh_cell_ids=True "
                "(so neighbor IDs are stored).")

        # Build an index for fast cell lookup
        perturb_index = _build_perturb_index(perturb_df)

        # Use partial so the dataset mapper sees only one argument as expected
        perturb_fn = partial(
            _perturb_batch_with_idx,
            index=perturb_index,
            seq_len_cell=seq_len_cell,
            pad_gene_tokens=pad_gene_tokens,
            log1p_values=log1p_values,
            adjust_positions=adjust_positions,
            n_segments=n_segments,
        )

    # Compute which cells the perturbations actually edit *before* mapping, from
    # the original dataset (knockout zeroes the gene tokens, so presence must be
    # read pre-perturbation). Row order is preserved by map, so these indices
    # stay valid for the perturbed dataset.
    perturbation_masks = None
    if return_only_perturbed_cells or return_perturbation_flags:
        _affected = _affected_cell_indices(
            dataset, perturb_df, seq_len_cell,
            return_masks=return_perturbation_flags)
        if return_perturbation_flags:
            affected_idx, perturbation_masks = _affected
        else:
            affected_idx = _affected
        keep_idx = affected_idx if return_only_perturbed_cells else None
    else:
        keep_idx = None

    # The neighbor-ID column (`cell_ids`) is large and never modified; re-encoding
    # it through .map is what makes the map stall. Keep it out of the map and
    # re-attach the original column afterwards (map preserves row order, so this
    # is a zero-copy column concat). Only neighborhood-on-specific-cells needs it
    # as *input*; otherwise drop it before mapping too so it isn't even decoded.
    has_cell_ids = "cell_ids" in dataset.column_names
    cell_ids_col = (dataset.select_columns(["cell_ids"]).with_format(None)
                    if has_cell_ids else None)
    map_input = (dataset if (needs_cell_ids_input or not has_cell_ids)
                 else dataset.remove_columns("cell_ids"))

    # Map over the *unformatted* dataset and restore the format afterwards.
    # Applying the dataset's torch format to the large nested token columns
    # inside .map can stall indefinitely; the perturbation fn reads raw data and
    # converts the edited columns to numpy itself.
    saved_format = dataset.format
    # On many-core nodes torch over-subscribes CPU threads, and the batch-wide
    # tensor ops in the perturbation fn (nonzero / elementwise) can stall. Cap
    # intra-op threads to 1 for the duration of the map and restore afterwards.
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        dataset = map_input.with_format(None).map(
            perturb_fn,
            batched=True,
            batch_size=batch_size,
            num_proc=nproc,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=False,
            remove_columns=["cell_ids"] if needs_cell_ids_input else None)
    finally:
        torch.set_num_threads(n_threads)
    if has_cell_ids:
        dataset = concatenate_datasets([dataset, cell_ids_col], axis=1)

    # Optionally attach per-perturbation boolean flag columns (one per perturb_df
    # row). Attached here as a zero-copy column concat -- never inside .map, which
    # would force datasets to re-encode every column.
    flag_cols = []
    if return_perturbation_flags:
        flag_cols = [f"pert_flag_idx_{i}"
                     for i in range(perturbation_masks.shape[1])]
        flags_ds = Dataset.from_dict(
            {name: perturbation_masks[:, i].tolist()
             for i, name in enumerate(flag_cols)})
        dataset = concatenate_datasets([dataset, flags_ds], axis=1)

    if saved_format.get("type") is not None:
        fmt_columns = saved_format["columns"]
        # If the saved format lists explicit columns, keep the new flag columns
        # in that format too (so their element access matches the other columns).
        if flag_cols and fmt_columns is not None:
            fmt_columns = list(fmt_columns) + flag_cols
        dataset.set_format(
            type=saved_format["type"],
            columns=fmt_columns,
            output_all_columns=saved_format["output_all_columns"],
            **saved_format.get("format_kwargs", {}))

    # Optionally, return the union of cells eligible before any edits.
    if return_only_perturbed_cells:
        dataset = dataset.select(keep_idx)

    return dataset
