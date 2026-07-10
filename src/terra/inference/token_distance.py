import logging
import pickle
import sys
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
try:
    from geomloss import SamplesLoss
except ImportError:  # pragma: no cover - optional dependency
    SamplesLoss = None

from terra.utils.helper import init_model, load_checkpoint
from terra.datasets.cell_datasets import init_cell_dataset
from terra.datasets.dataloaders import init_dataloader_and_sampler
from terra.masks.block_masking import BlockMaskCollator


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def _geomloss_distance_pointcloud(
    X: np.ndarray,
    Y: np.ndarray,
    loss: str,
    p: int | None = None,
    blur: float = 0.01,
    backend: str = "tensorized",
    device: str | None = None,
) -> float:
    """
    Compute a distribution distance between two point clouds using GeomLoss.

    Parameters
    ----------
    X:
        Array of shape (n_points, n_dims).
    Y:
        Array of shape (n_points, n_dims).
    loss:
        GeomLoss distance type.
    p:
        Cost exponent for Sinkhorn.
    blur:
        GeomLoss blur parameter.
    backend:
        GeomLoss backend.
    device:
        Optional device string used for the GeomLoss computation.

    Returns
    -------
    float
        Scalar point-cloud distance.
    """
    if SamplesLoss is None:
        raise ImportError(
            "geomloss is required for infer_token_distance(). "
            "Install it to use point-cloud distances."
        )

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"X and Y must be 2D arrays. Got {X.shape}, {Y.shape}.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            "X and Y must have the same dimensionality. "
            f"Got {X.shape[1]} and {Y.shape[1]}."
        )

    tX = torch.from_numpy(X)
    tY = torch.from_numpy(Y)
    if device is not None:
        tX = tX.to(device)
        tY = tY.to(device)

    if loss == "sinkhorn":
        if p not in (1, 2):
            raise ValueError("For loss='sinkhorn', p must be 1 or 2.")
        loss_fn = SamplesLoss(loss="sinkhorn", p=p, blur=blur, backend=backend)
    elif loss == "energy":
        loss_fn = SamplesLoss(loss="energy", blur=blur, backend=backend)
    elif loss == "gaussian":
        loss_fn = SamplesLoss(loss="gaussian", blur=blur, backend=backend)
    else:
        raise ValueError("loss must be one of {'sinkhorn','energy','gaussian'}.")

    val = loss_fn(tX, tY)
    return float(val.detach().cpu().item())


def _init_embed_inference_loader(
    dataset: Dataset,
    model_config: dict,
    vocab_size: int,
    batch_size: int,
    pin_memory: bool,
    num_workers: int,
    n_special_tokens: int,
):
    """
    Initialize the embedding dataloader used by token-level inference utilities.

    Mirrors the loader-building block in ``embed.py`` (v9), including the
    ``special_token_pad_ratio`` argument of ``BlockMaskCollator``.
    """
    mask_collator = BlockMaskCollator(
        n_targets=model_config['mask']['n_targets'],
        n_contexts=model_config['mask']['n_contexts'],
        n_segments=model_config['data']['n_segments'],
        seq_len_cell=model_config['data']['seq_len_cell'],
        seq_len_neighborhood=model_config['data']['seq_len_neighborhood'],
        n_special_tokens=n_special_tokens,
        per_block_mask_ratio=model_config['mask']['per_block_mask_ratio'],
        sample_segments=False,
        sample_gene_masks=False,
        restrict_special_attention=model_config['meta']['restrict_special_attention'],
        special_token_pad_ratio=1.0)

    cell_dataset = init_cell_dataset(
        dataset=dataset,
        vocab_size=vocab_size,
        seq_len_cell=model_config['data']['seq_len_cell'],
        seq_len_neighborhood=model_config['data']['seq_len_neighborhood'],
        tokenizer_type=model_config['data']['tokenizer_type'],
        gt_type=model_config['meta']['gt_type'],
        cell_pos_enc=model_config['meta']['cell_pos_enc'],
        special_tokens=model_config['meta']['special_tokens'],
        sampling_strategy=None,
        n_nonzero_tokens_list=[],
        include_cell_id=True,
        sep_gene_tokens_neb=model_config['data']['sep_gene_tokens_neb'])

    loader, _ = init_dataloader_and_sampler(
        cell_dataset=cell_dataset,
        batch_size=batch_size,
        distributed=False,
        world_size=1,
        rank=0,
        collate_fn=mask_collator,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=False,
        mega_batch_mult_max=model_config['data'].get('mega_batch_mult_max', 1000))
    return loader


def _resolve_nonpad_token_mask(
    batch_dict: dict,
    ns_tokens: torch.Tensor,
    n_special_tokens: int,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Resolve a boolean non-padding mask for non-special tokens.
    """
    for mask_key in ("padding_mask", "attention_mask"):
        if mask_key not in batch_dict:
            continue
        mask = batch_dict[mask_key]
        if not isinstance(mask, torch.Tensor):
            continue
        if mask.shape[-1] == ns_tokens.shape[1] + n_special_tokens:
            return mask[:, n_special_tokens:].bool()
        if mask.shape[-1] == ns_tokens.shape[1]:
            return mask.bool()
    return ns_tokens != pad_token_id


@torch.inference_mode()
def infer_token_distance(
    dataset_original: Dataset,
    dataset_perturbed: Dataset,
    model_folder_path: str,
    emb_layer: int | None = None,
    batch_size: int = 128,
    pin_memory: bool = False,
    num_workers: int = 12,
    loss: Literal["sinkhorn", "energy", "gaussian"] = "sinkhorn",
    p: int | None = 2,
    blur: float = 0.01,
    backend: str = "tensorized",
    device: str | None = None,
    ignore_spc_tokens: bool = True,
) -> dict:
    """
    Compare token-embedding point clouds between aligned original and perturbed
    datasets and return per-cell distances.

    Parameters
    -----------
    dataset_original:
        Tokenized huggingface dataset for the original cells.
    dataset_perturbed:
        Tokenized huggingface dataset for the perturbed cells. The order must
        match `dataset_original`.
    model_folder_path:
        Path to the folder containing the model config, token dictionary, and
        model checkpoint.
    emb_layer:
        Layer for which to retrieve the embedding.
    batch_size:
        Dataloader param.
    pin_memory:
        Dataloader param.
    num_workers:
        Number of workers used.
    loss:
        GeomLoss distance type. One of `sinkhorn`, `energy`, or `gaussian`.
    p:
        Cost exponent for Sinkhorn. Must be 1 or 2 when `loss='sinkhorn'`.
    blur:
        GeomLoss blur parameter.
    backend:
        GeomLoss backend.
    device:
        Optional device string used for the GeomLoss computation. Inference
        still follows the repository's default device selection.
    ignore_spc_tokens:
        Passed through to the model embedding extraction path.

    Returns
    -----------
    output_score : dict
        Dictionary with one entry per token-cloud distance plus run metadata:

        - ``cell_score``: numpy array of shape ``(n_cells,)`` with the distance
          between the original and perturbed cell-only embeddings restricted to
          the cell's own tokens. ``NaN`` for cells whose original or perturbed
          token set is empty.
        - ``spatial_cell_score``: numpy array of shape ``(n_cells,)`` with the
          distance between the original and perturbed full-context (spatially
          contextualized) embeddings restricted to the cell's own tokens.
        - ``neighborhood_score``: numpy array of shape ``(n_cells,)`` with the
          distance between the original and perturbed full-context embeddings
          over all valid (non-padding) tokens.
        - ``meta``: dictionary recording the distance settings (``loss``,
          ``p``, ``blur``, ``backend``, ``device``) and
          ``empty_token_set_counts``, the per-score counts of cells skipped
          because the original, perturbed, or either token set was empty.
    """
    if len(dataset_original) != len(dataset_perturbed):
        raise ValueError(
            "dataset_original and dataset_perturbed must have the same length. "
            f"Got {len(dataset_original)} and {len(dataset_perturbed)}."
        )
    if loss == "sinkhorn" and p not in (1, 2):
        raise ValueError("For loss='sinkhorn', p must be 1 or 2.")

    logger.info('STEP 1: LOADING CONFIG...')
    model_config_file_path = Path(model_folder_path) / 'model_config.yaml'
    token_dictionary_file_path = Path(model_folder_path) / 'token_dictionary.pkl'
    model_checkpoint_path = Path(model_folder_path) / 'model_checkpoint.pt'

    with open(model_config_file_path, 'r') as file:
        model_config = yaml.safe_load(file)

    n_special_tokens = len(model_config['meta']['special_tokens'])
    seq_len = (
        model_config['data']['seq_len_cell'] +
        model_config['data']['seq_len_neighborhood'] +
        n_special_tokens)
    seq_len_cell = model_config['data']['seq_len_cell']
    pad_token_id = 0

    if emb_layer is None:
        emb_layer = model_config['meta']['enc_depth']

    with open(token_dictionary_file_path, 'rb') as file:
        token_dict = pickle.load(file)
    vocab_size = len(token_dict)
    n_special_values = model_config['data'].get('n_special_values', 0)

    logger.info('STEP 2: GENERATING TOKEN DISTANCES...')
    if not torch.cuda.is_available():
        inference_device = torch.device('cpu')
    else:
        inference_device = torch.device('cuda:0')
        torch.cuda.set_device(inference_device)

    geomloss_device = device
    if geomloss_device is None:
        geomloss_device = (
            str(inference_device) if inference_device.type == 'cuda' else 'cpu'
        )

    target_encoder, _ = init_model(
        gt_type=model_config['meta']['gt_type'],
        count_encoding=model_config['meta']['count_encoding'],
        n_value_bins=model_config['meta']['n_value_bins'],
        cell_pos_enc=model_config['meta']['cell_pos_enc'],
        device=inference_device,
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
        mlp_ratio=model_config['meta']['mlp_ratio'],
        use_flash_attention=model_config['meta']['use_flash_attention'],
        api_version=model_config['meta']['api_version'],
        sep_gene_tokens_neb=model_config['data']['sep_gene_tokens_neb'],
        predict_gene=model_config['meta']['predict_gene'],
        pos_learnable=model_config['meta']['pos_learnable'],
        nz_spc=model_config['data'].get('nz_spc', False),
        mlp_bias=model_config['meta'].get('mlp_bias', True))

    if model_config['meta']['api_version'] != 'v3':
        return_layer_emb_fn = target_encoder.return_layer_emb
    else:
        return_layer_emb_fn = target_encoder.backbone.return_layer_emb

    loader_original = _init_embed_inference_loader(
        dataset=dataset_original,
        model_config=model_config,
        vocab_size=vocab_size,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        n_special_tokens=n_special_tokens,
    )
    loader_perturbed = _init_embed_inference_loader(
        dataset=dataset_perturbed,
        model_config=model_config,
        vocab_size=vocab_size,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        n_special_tokens=n_special_tokens,
    )

    _, _, target_encoder, _, _, _, _ = load_checkpoint(
            device=inference_device,
            r_path=model_checkpoint_path,
            encoder=None,
            predictor=None,
            target_encoder=target_encoder,
            opt=None,
            scaler=None,
            is_training=False)
    target_encoder.eval()

    n_cells = len(dataset_original)
    cell_score = np.full(n_cells, np.nan, dtype=np.float32)
    spatial_cell_score = np.full(n_cells, np.nan, dtype=np.float32)
    neighborhood_score = np.full(n_cells, np.nan, dtype=np.float32)
    empty_counts = {
        "cell_score": {"original": 0, "perturbed": 0, "either": 0},
        "spatial_cell_score": {"original": 0, "perturbed": 0, "either": 0},
        "neighborhood_score": {"original": 0, "perturbed": 0, "either": 0},
    }

    write_idx = 0
    for batch_original, batch_perturbed in tqdm(
        zip(loader_original, loader_perturbed),
        total=len(loader_original),
    ):
        udata_original, _, _, masks_attention_original, _ = batch_original
        udata_perturbed, _, _, masks_attention_perturbed, _ = batch_perturbed

        batch_cell_ids_original = list(udata_original.get('cell_id', []))
        batch_cell_ids_perturbed = list(udata_perturbed.get('cell_id', []))
        if batch_cell_ids_original and batch_cell_ids_perturbed:
            if batch_cell_ids_original != batch_cell_ids_perturbed:
                raise ValueError(
                    "dataset_original and dataset_perturbed are not aligned. "
                    "Encountered mismatched cell_id ordering within a batch."
                )

        for key in udata_original.keys():
            if key != 'cell_id':
                udata_original[key] = udata_original[key].to(
                    inference_device, non_blocking=True)
        for key in udata_perturbed.keys():
            if key != 'cell_id':
                udata_perturbed[key] = udata_perturbed[key].to(
                    inference_device, non_blocking=True)
        masks_attention_original = masks_attention_original.to(
            inference_device, non_blocking=True)
        masks_attention_perturbed = masks_attention_perturbed.to(
            inference_device, non_blocking=True)

        ns_tokens_original = udata_original['tokens'][:, n_special_tokens:]
        ns_tokens_perturbed = udata_perturbed['tokens'][:, n_special_tokens:]

        with torch.cuda.amp.autocast(
            dtype=torch.bfloat16,
            enabled=model_config['meta']['use_bfloat16']):
            emb_layers = [emb_layer]

            full_ctx_original, cell_only_ctx_original = return_layer_emb_fn(
                layers=emb_layers,
                batch=udata_original,
                masks_attention=masks_attention_original,
                need_cell_only_context=True,
                ignore_spc_tokens=ignore_spc_tokens,
            )
            full_ctx_perturbed, cell_only_ctx_perturbed = return_layer_emb_fn(
                layers=emb_layers,
                batch=udata_perturbed,
                masks_attention=masks_attention_perturbed,
                need_cell_only_context=True,
                ignore_spc_tokens=ignore_spc_tokens,
            )

        c_emb_original = cell_only_ctx_original[emb_layer].detach().cpu()
        c_emb_perturbed = cell_only_ctx_perturbed[emb_layer].detach().cpu()
        n_emb_original = full_ctx_original[emb_layer].detach().cpu()
        n_emb_perturbed = full_ctx_perturbed[emb_layer].detach().cpu()
        ns_tokens_original = ns_tokens_original.detach().cpu()
        ns_tokens_perturbed = ns_tokens_perturbed.detach().cpu()

        valid_mask_original = _resolve_nonpad_token_mask(
            batch_dict=udata_original,
            ns_tokens=ns_tokens_original,
            n_special_tokens=n_special_tokens,
            pad_token_id=pad_token_id,
        ).detach().cpu()
        valid_mask_perturbed = _resolve_nonpad_token_mask(
            batch_dict=udata_perturbed,
            ns_tokens=ns_tokens_perturbed,
            n_special_tokens=n_special_tokens,
            pad_token_id=pad_token_id,
        ).detach().cpu()

        cell_mask_original = valid_mask_original.clone()
        cell_mask_original[:, seq_len_cell:] = False
        cell_mask_perturbed = valid_mask_perturbed.clone()
        cell_mask_perturbed[:, seq_len_cell:] = False

        batch_size_actual = c_emb_original.shape[0]
        for i in range(batch_size_actual):
            cell_idx = write_idx + i

            X_cell_original = c_emb_original[i][cell_mask_original[i]].numpy()
            X_cell_perturbed = c_emb_perturbed[i][cell_mask_perturbed[i]].numpy()
            if X_cell_original.shape[0] == 0:
                empty_counts["cell_score"]["original"] += 1
            if X_cell_perturbed.shape[0] == 0:
                empty_counts["cell_score"]["perturbed"] += 1
            if X_cell_original.shape[0] == 0 or X_cell_perturbed.shape[0] == 0:
                empty_counts["cell_score"]["either"] += 1
            else:
                cell_score[cell_idx] = _geomloss_distance_pointcloud(
                    X_cell_original,
                    X_cell_perturbed,
                    loss=loss,
                    p=p,
                    blur=blur,
                    backend=backend,
                    device=geomloss_device,
                )

            X_spatial_original = n_emb_original[i][cell_mask_original[i]].numpy()
            X_spatial_perturbed = n_emb_perturbed[i][cell_mask_perturbed[i]].numpy()
            if X_spatial_original.shape[0] == 0:
                empty_counts["spatial_cell_score"]["original"] += 1
            if X_spatial_perturbed.shape[0] == 0:
                empty_counts["spatial_cell_score"]["perturbed"] += 1
            if X_spatial_original.shape[0] == 0 or X_spatial_perturbed.shape[0] == 0:
                empty_counts["spatial_cell_score"]["either"] += 1
            else:
                spatial_cell_score[cell_idx] = _geomloss_distance_pointcloud(
                    X_spatial_original,
                    X_spatial_perturbed,
                    loss=loss,
                    p=p,
                    blur=blur,
                    backend=backend,
                    device=geomloss_device,
                )

            X_neighborhood_original = n_emb_original[i][valid_mask_original[i]].numpy()
            X_neighborhood_perturbed = n_emb_perturbed[i][valid_mask_perturbed[i]].numpy()
            if X_neighborhood_original.shape[0] == 0:
                empty_counts["neighborhood_score"]["original"] += 1
            if X_neighborhood_perturbed.shape[0] == 0:
                empty_counts["neighborhood_score"]["perturbed"] += 1
            if (
                X_neighborhood_original.shape[0] == 0
                or X_neighborhood_perturbed.shape[0] == 0
            ):
                empty_counts["neighborhood_score"]["either"] += 1
            else:
                neighborhood_score[cell_idx] = _geomloss_distance_pointcloud(
                    X_neighborhood_original,
                    X_neighborhood_perturbed,
                    loss=loss,
                    p=p,
                    blur=blur,
                    backend=backend,
                    device=geomloss_device,
                )

        write_idx += batch_size_actual

    if write_idx != n_cells:
        raise RuntimeError(
            "Processed cell count does not match dataset length. "
            f"Processed {write_idx} cells for a dataset of length {n_cells}."
        )

    return {
        "cell_score": cell_score,
        "spatial_cell_score": spatial_cell_score,
        "neighborhood_score": neighborhood_score,
        "meta": {
            "loss": loss,
            "p": p,
            "blur": blur,
            "backend": backend,
            "device": geomloss_device,
            "empty_token_set_counts": empty_counts,
        },
    }
