"""Population-level perturbation scoring.

These utilities quantify a perturbation's effect at the level of a *cell
population* (a group of cells sharing a label, e.g. a niche or cell type).
Each cell is represented by a single pooled embedding in ``adata.obsm`` (for
example ``cell_emb`` / ``spatial_cell_emb`` / ``neighborhood_emb``). For every
label group the set of unperturbed embeddings and the set of perturbed
embeddings are treated as two empirical distributions, and the distance between
them is computed with GeomLoss.

This complements :func:`terra.inference.infer_token_distance`, which scores the
*per-cell* token-embedding clouds. The two operate on different objects (pooled
per-cell vectors here vs. per-token vectors there) and are not directly
comparable in absolute scale.

The point-cloud distance itself is shared with ``token_distance`` via
``_geomloss_distance_pointcloud`` so there is a single implementation.
"""

import numpy as np
import pandas as pd

from .token_distance import _geomloss_distance_pointcloud


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row (sample) of ``X``."""
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)


def _agg_fn_from_name(agg):
    """Resolve an aggregation name (``'min'``/``'mean'``/``'median'``) or callable."""
    if isinstance(agg, str):
        agg_name = agg.lower()
        if agg_name == "min":
            return np.min
        if agg_name == "mean":
            return np.mean
        if agg_name == "median":
            return np.median
        raise ValueError("agg must be 'min', 'mean', 'median', or a callable")
    if callable(agg):
        return agg
    raise ValueError("agg must be 'min', 'mean', 'median', or a callable")


def cosine_sim_per_row(A, B, eps=1e-12):
    """
    Row-wise cosine similarity between ``A`` and ``B`` (shape ``[n, d]``).

    Handles zero-norm rows safely via ``eps``. Unlike the point-cloud distances
    below, this is a *matched* comparison: row ``i`` of ``A`` is compared to row
    ``i`` of ``B`` (e.g. a cell's unperturbed vs. perturbed embedding).
    """
    A = np.asarray(A)
    B = np.asarray(B)

    A_norm = np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = np.linalg.norm(B, axis=1, keepdims=True)

    A_unit = A / np.maximum(A_norm, eps)
    B_unit = B / np.maximum(B_norm, eps)

    return np.sum(A_unit * B_unit, axis=1)


def summarize_cosine_sim_by_label(
    adata,
    label_key,
    labels=None,
    pairs=None,
    agg="mean",
    sort_by=None,
    ascending=True,
    eps=1e-12,
    ignore_zeros=False,
):
    """
    Aggregate row-wise cosine similarity between embedding pairs per label.

    For each ``(A_key, B_key, out_col)`` in ``pairs`` the row-wise cosine
    similarity between ``adata.obsm[A_key]`` (unperturbed) and
    ``adata.obsm[B_key]`` (perturbed) is aggregated within each label group of
    ``adata.obs[label_key]`` and returned as a sorted DataFrame. Values near 1
    indicate little change.

    Parameters
    ----------
    ignore_zeros:
        If True, aggregate only over nonzero cosine similarities.
    """
    if pairs is None:
        pairs = [
            ("cell_emb", "cell_emb_perturb_case1", "cell_emb_cos_sim"),
            ("spatial_cell_emb", "spatial_cell_emb_perturb_case1", "spatial_cell_emb_cos_sim"),
            ("neighborhood_emb", "neighborhood_emb_perturb_case1", "neighborhood_emb_cos_sim"),
        ]
    if labels is None:
        labels = sorted(pd.unique(adata.obs[label_key]))

    agg_fn = _agg_fn_from_name(agg)

    rows = []
    for label in labels:
        adata_sub = adata[adata.obs[label_key] == label]
        row = {"label": label, "n_cells": int(adata_sub.n_obs)}

        for A_key, B_key, out_col in pairs:
            sims = cosine_sim_per_row(
                adata_sub.obsm[A_key],
                adata_sub.obsm[B_key],
                eps=eps,
            )

            if ignore_zeros:
                sims = sims[sims != 0]

            row[out_col] = float("nan") if sims.size == 0 else float(agg_fn(sims))

        rows.append(row)

    df = pd.DataFrame(rows)
    if sort_by is not None:
        df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
    return df


def summarize_w1_by_label(
    adata,
    label_key,
    labels=None,
    pairs=None,
    agg="mean",
    sort_by=None,
    ascending=True,
    eps=1e-12,
    ignore_zeros=False,
    blur=0.01,
    backend="tensorized",
    device=None,
    threshold=None,
):
    """
    W1 (Sinkhorn, ``p=1``) between point-cloud distributions per label.

    For each label group and each ``(A_key, B_key, out_col)`` pair, the sets of
    L2-normalized per-cell embeddings under the unperturbed (``A_key``) and
    perturbed (``B_key``) conditions are compared as two empirical
    distributions. Returns a DataFrame ``["label", "n_cells", out_col...]``,
    optionally sorted.

    Parameters
    ----------
    threshold:
        If set and a group has more than ``threshold`` cells, both clouds are
        randomly subsampled to ``threshold`` rows (same indices) before the
        distance is computed.
    """
    if pairs is None:
        pairs = [
            ("cell_emb", "cell_emb_perturb_case1", "cell_emb_w1"),
            ("spatial_cell_emb", "spatial_cell_emb_perturb_case1", "spatial_cell_emb_w1"),
            ("neighborhood_emb", "neighborhood_emb_perturb_case1", "neighborhood_emb_w1"),
        ]
    return _summarize_pointcloud_by_label(
        adata, label_key, labels, pairs, agg, sort_by, ascending, eps,
        ignore_zeros, blur, backend, device, threshold,
        loss="sinkhorn", p=1,
    )


def summarize_w2_by_label(
    adata,
    label_key,
    labels=None,
    pairs=None,
    agg="mean",
    sort_by=None,
    ascending=True,
    eps=1e-12,
    ignore_zeros=False,
    blur=0.01,
    backend="tensorized",
    device=None,
    threshold=None,
):
    """
    W2 (Sinkhorn, ``p=2``) between point-cloud distributions per label.

    L2-normalizes rows before computing. See :func:`summarize_w1_by_label` for
    the shared parameter semantics.
    """
    if pairs is None:
        pairs = [
            ("cell_emb", "cell_emb_perturb_case1", "cell_emb_w2"),
            ("spatial_cell_emb", "spatial_cell_emb_perturb_case1", "spatial_cell_emb_w2"),
            ("neighborhood_emb", "neighborhood_emb_perturb_case1", "neighborhood_emb_w2"),
        ]
    return _summarize_pointcloud_by_label(
        adata, label_key, labels, pairs, agg, sort_by, ascending, eps,
        ignore_zeros, blur, backend, device, threshold,
        loss="sinkhorn", p=2,
    )


def summarize_energy_by_label(
    adata,
    label_key,
    labels=None,
    pairs=None,
    agg="mean",
    sort_by=None,
    ascending=True,
    eps=1e-12,
    ignore_zeros=False,
    blur=0.5,
    backend="tensorized",
    device=None,
    threshold=None,
):
    """
    Energy distance (GeomLoss ``'energy'``) between point-cloud distributions
    per label. L2-normalizes rows before computing.
    """
    if pairs is None:
        pairs = [
            ("cell_emb", "cell_emb_perturb_case1", "cell_emb_energy"),
            ("spatial_cell_emb", "spatial_cell_emb_perturb_case1", "spatial_cell_emb_energy"),
            ("neighborhood_emb", "neighborhood_emb_perturb_case1", "neighborhood_emb_energy"),
        ]
    return _summarize_pointcloud_by_label(
        adata, label_key, labels, pairs, agg, sort_by, ascending, eps,
        ignore_zeros, blur, backend, device, threshold,
        loss="energy", p=None,
    )


def summarize_mmd_by_label(
    adata,
    label_key,
    labels=None,
    pairs=None,
    agg="mean",
    sort_by=None,
    ascending=True,
    eps=1e-12,
    ignore_zeros=False,
    blur=0.5,
    backend="tensorized",
    device=None,
    threshold=None,
):
    """
    Gaussian MMD-like loss (GeomLoss ``'gaussian'``) between point-cloud
    distributions per label. L2-normalizes rows before computing.
    """
    if pairs is None:
        pairs = [
            ("cell_emb", "cell_emb_perturb_case1", "cell_emb_mmd"),
            ("spatial_cell_emb", "spatial_cell_emb_perturb_case1", "spatial_cell_emb_mmd"),
            ("neighborhood_emb", "neighborhood_emb_perturb_case1", "neighborhood_emb_mmd"),
        ]
    return _summarize_pointcloud_by_label(
        adata, label_key, labels, pairs, agg, sort_by, ascending, eps,
        ignore_zeros, blur, backend, device, threshold,
        loss="gaussian", p=None,
    )


def _summarize_pointcloud_by_label(
    adata,
    label_key,
    labels,
    pairs,
    agg,
    sort_by,
    ascending,
    eps,
    ignore_zeros,
    blur,
    backend,
    device,
    threshold,
    loss,
    p,
):
    """Shared driver for the point-cloud (Sinkhorn/energy/gaussian) summaries."""
    if labels is None:
        labels = sorted(pd.unique(adata.obs[label_key]))

    agg_fn = _agg_fn_from_name(agg)

    rows = []
    for label in labels:
        adata_sub = adata[adata.obs[label_key] == label]
        row = {"label": label, "n_cells": int(adata_sub.n_obs)}

        for A_key, B_key, out_col in pairs:
            XA = _l2_normalize_rows(np.asarray(adata_sub.obsm[A_key]), eps=eps)
            XB = _l2_normalize_rows(np.asarray(adata_sub.obsm[B_key]), eps=eps)

            if threshold is not None and XA.shape[0] > threshold:
                idx = np.random.choice(XA.shape[0], size=int(threshold), replace=False)
                XA = XA[idx]
                XB = XB[idx]

            d = _geomloss_distance_pointcloud(
                XA, XB, loss=loss, p=p, blur=blur, backend=backend, device=device
            )

            vals = np.array([d], dtype=float)
            if ignore_zeros:
                vals = vals[vals != 0]

            row[out_col] = float("nan") if vals.size == 0 else float(agg_fn(vals))
        rows.append(row)

    df = pd.DataFrame(rows)
    if sort_by is not None:
        df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
    return df
