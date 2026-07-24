"""
LoRA / selective-unfreeze finetuning for TERRA classification.

Standalone usage (reads data paths from config):
    python src/app/finetune.py --fname reproducibility/config/finetuning/config.yaml

Programmatic usage from a cross-validation loop:
    from terra.training.finetune import finetune

    # Inner fold — early stopping on val, returns val accuracy for LR selection:
    predictions, targets, accuracy = finetune(
        args=args,
        train_adata=train_adata, train_dataset=train_dataset,
        val_adata=val_adata, val_dataset=val_dataset,
    )

    # Outer fold — early stopping on val, final eval on held-out test:
    predictions, targets, accuracy = finetune(
        args=args,
        train_adata=train_adata, train_dataset=train_dataset,
        val_adata=val_adata, val_dataset=val_dataset,
        test_adata=test_adata, test_dataset=test_dataset,
        save_folder_path=output_dir,
    )
"""
from __future__ import annotations

import os
import sys
import argparse
import pickle
import logging
import yaml
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from datasets import Dataset, load_from_disk

from terra.utils.helper import (init_model, load_checkpoint, apply_peft,
                                parse_protein_init_kwargs)
from terra.datasets.cell_datasets import init_cell_dataset
from terra.datasets.dataloaders import init_dataloader_and_sampler
from terra.masks.block_masking  import BlockMaskCollator
from terra.masks.cell_masking import CellMaskCollator
from terra.utils.distributed import init_distributed
from terra.models.modules import ClassificationModel
from terra.utils.embedding import create_binary_selection_mask


os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1" # Better error propagation

_GLOBAL_SEED = 0
LOCAL_RANK = None

logger = logging.getLogger(__name__)


# Setup argument parsing
def parse_arguments():
    """
    Parse config file name from command-line arguments and return hyperparameters in a nested dictionary.
    """
    parser = argparse.ArgumentParser(
        description='Run TERRA finetuning.')
    parser.add_argument(
        '--fname',
        type=str,
        default='configs.yaml',
        help='Name of the config file to load.'
    )

    parser_args = parser.parse_args()

    # Get the config file name from command line argument
    args_fname = parser_args.fname

    # Read parameters from config file
    with open(args_fname, "r") as f:
        args = yaml.safe_load(f)

    return args


@torch.no_grad()
def validate(
        model,
        val_loader,
        val_label_lookup,
        criterion,
        device,
        n_special_tokens,
        model_config,
        selection_type,
        agg_excluded_tokens,
        top_k
    ):
    """
    Run one validation pass and return loss and accuracy scalars.

    Called at the end of each training epoch to drive early stopping: the
    returned loss is compared against the best seen so far, and the patience
    counter is incremented or reset accordingly. Because it is called inside
    the training loop, it restores model.train() before returning so that the
    next epoch begins in the correct mode.

    Returns scalars only — not per-sample predictions. Use test() after
    training is complete if you need per-sample outputs for a classification
    report.

    Parameters
    ----------
    model:
        The model being trained.
    val_loader:
        Dataloader for the validation split.
    val_label_lookup:
        Pandas Series indexed by cell_id mapping to integer class labels.
    criterion:
        Loss function (CrossEntropyLoss).
    device:
        Device tensors are moved to.
    n_special_tokens:
        Number of leading special tokens to skip when building the selection mask.
    model_config:
        Model configuration dict loaded from the pretrained checkpoint.
    selection_type:
        'agg_cell' or 'agg_graph' — determines which tokens are pooled.
    agg_excluded_tokens:
        Token ids to exclude from aggregation (or None).
    top_k:
        If set, restrict aggregation to the top-k expressed genes.

    Returns
    -------
    val_loss:
        Mean cross-entropy loss over the validation set.
    val_accuracy:
        Fraction of correctly classified cells.
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # for udata, _, _, masks_attention, _ in val_loader:
    for itr, (udata, _, _, masks_attention, _) in tqdm(enumerate(val_loader), desc=f"Validation"):
        for key in udata.keys():
            if key != 'cell_id':
                udata[key] = udata[key].to(device, non_blocking=True)
        masks_attention = masks_attention.to(device, non_blocking=True)

        ns_tokens = udata['tokens'][:, n_special_tokens:]
        # Create cell mask or neighborhood mask for aggregation depending on the selection type (agg_cell or agg_graph)
        if selection_type == 'agg_cell':
            selection_mask = create_binary_selection_mask(
                ns_tokens,
                selection_type=selection_type,
                excluded_tokens=agg_excluded_tokens,
                seq_len_cell=model_config['data']['seq_len_cell'],
                top_k=top_k,
            )
        elif selection_type == 'agg_graph':
            selection_mask = create_binary_selection_mask(
                ns_tokens,
                selection_type=selection_type,
                excluded_tokens=agg_excluded_tokens,
                seq_len_cell=model_config['data']['seq_len_cell'],
                top_k=top_k,
                n_segments=model_config['data']['n_segments']
            )
        else:
            raise ValueError(f"Selection type {selection_type} not supported.")

        logits = model(
            udata=udata,
            masks_attention=masks_attention,
            selection_mask=selection_mask,
        )

        labels = torch.tensor(
            val_label_lookup[udata['cell_id']].values,
            dtype=torch.long,
            device=device
        )

        loss = criterion(logits, labels)
        val_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        val_correct += (preds == labels).sum().item()
        val_total += labels.size(0)

        # if itr > 5:
        #     break  # For debugging, remove this line for full validation

    model.train()

    return val_loss / len(val_loader), val_correct / val_total


@torch.no_grad()
def test(
        model,
        loader,
        label_lookup,
        device,
        n_special_tokens,
        model_config,
        selection_type,
        agg_excluded_tokens,
        top_k
    ) -> Tuple[List[int], List[int], float]:
    """
    Run a final evaluation pass and return per-sample predictions.

    Called once after training is complete on the held-out evaluation set
    (val for inner folds, test for outer folds). Unlike validate(), this
    returns per-sample predictions and targets so that the caller can write
    a full classification report (per-class precision/recall/F1). It does
    not compute loss and does not restore training mode.

    Returns
    -------
    predictions:
        List of predicted class indices, one per cell.
    targets:
        List of ground truth class indices, one per cell.
    accuracy:
        Fraction of correctly classified cells.
    """
    model.eval()
    all_preds = []
    all_targets = []

    # for udata, _, _, masks_attention, _ in loader:
    for itr, (udata, _, _, masks_attention, _) in tqdm(enumerate(loader), desc=f"Test"):
        for key in udata.keys():
            if key != 'cell_id':
                udata[key] = udata[key].to(device, non_blocking=True)
        masks_attention = masks_attention.to(device, non_blocking=True)

        ns_tokens = udata['tokens'][:, n_special_tokens:]
        # Create cell mask or neighborhood mask for aggregation depending on the selection type (agg_cell or agg_graph)
        if selection_type == 'agg_cell':
            selection_mask = create_binary_selection_mask(
                ns_tokens,
                selection_type=selection_type,
                excluded_tokens=agg_excluded_tokens,
                seq_len_cell=model_config['data']['seq_len_cell'],
                top_k=top_k,
            )
        elif selection_type == 'agg_graph':
            selection_mask = create_binary_selection_mask(
                ns_tokens,
                selection_type=selection_type,
                excluded_tokens=agg_excluded_tokens,
                seq_len_cell=model_config['data']['seq_len_cell'],
                top_k=top_k,
                n_segments=model_config['data']['n_segments']
            )
        else:
            raise ValueError(f"Selection type {selection_type} not supported.")

        logits = model(
            udata=udata,
            masks_attention=masks_attention,
            selection_mask=selection_mask,
        )

        labels = torch.tensor(
            label_lookup[udata['cell_id']].values,
            dtype=torch.long,
            device=device
        )

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())

        # if itr > 5:
            # break  # For debugging, remove this line for full evaluation

    accuracy = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)

    return all_preds, all_targets, accuracy


def finetune(
        args: dict,
        train_adata: AnnData,
        train_dataset: Dataset,
        val_adata: AnnData | None = None,
        val_dataset: Dataset | None = None,
        test_adata: AnnData | None = None,
        test_dataset: Dataset | None = None,
        save_folder_path: str | None = None,
        resume_checkpoint_path: str | None = None,
    ) -> Tuple[List[int], List[int], float]:
    """
    Finetune a pretrained TERRA encoder for niche/cell transfer.

    Loads the encoder from a saved checkpoint, applies LoRA adapters or
    selective weight unfreezing (controlled by args['finetune']['use_peft']),
    attaches a linear or MLP classification head, and trains end-to-end.

    Designed to be called from a cross-validation loop. The caller controls
    which data splits are provided:

    Inner fold (LR search):
        Pass val_adata/val_dataset only. Early stopping is applied on val loss;
        final evaluation is on val. The returned accuracy is used by the caller
        to select the best learning rate across inner folds.

    Outer fold (final training):
        Pass test_adata/test_dataset, and optionally val_adata/val_dataset for early stopping. If val is provided, training stops early when val loss stops improving;
        the best model state is restored before evaluating on test. If val is not provided, training runs to num_epochs. The returned
        predictions and accuracy are the per-fold test results.

    At least one of (val, test) must be provided.

    Parameters
    ----------
    args:
        Config dict. Relevant sub-dicts: args['model'], args['data'],
        args['finetune'].
    train_adata:
        AnnData with obs columns 'cell_id' and the integer label column.
    train_dataset:
        Pre-tokenized Hugging Face Dataset for training.
    val_adata:
        AnnData for the validation split (optional).
    val_dataset:
        Pre-tokenized Hugging Face Dataset for validation (optional).
    test_adata:
        AnnData for the held-out test split (optional).
    test_dataset:
        Pre-tokenized Hugging Face Dataset for the test split (optional).
    save_folder_path:
        Directory under which a timestamped run folder is created for
        checkpoints and params.yaml. If None, no artifacts are saved.
    resume_checkpoint_path:
        Path to a checkpoint file to resume from. The checkpoint must contain
        'model', 'optimizer', and 'epoch' keys.

    Returns
    -------
    predictions:
        Per-cell predicted class indices on the evaluation set.
    targets:
        Per-cell ground truth class indices on the evaluation set.
    accuracy:
        Fraction of correctly classified cells on the evaluation set.
    """
    # -------------------------------------------------------------------- #
    # VALIDATE INPUTS
    # -------------------------------------------------------------------- #
    has_val = val_adata is not None and val_dataset is not None
    has_test = test_adata is not None and test_dataset is not None

    if not has_val and not has_test:
        raise ValueError("At least one of (val, test) must be provided.")

    # Determine evaluation set
    if has_test:
        eval_adata, eval_dataset = test_adata, test_dataset
        eval_set_name = "Test"
    else:
        eval_adata, eval_dataset = val_adata, val_dataset
        eval_set_name = "Validation"

    # -------------------------------------------------------------------- #
    # BACKEND SETUP
    # -------------------------------------------------------------------- #
    logger.info("Configuring backend...")
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_GLOBAL_SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -------------------------------------------------------------------- #
    # LOAD MODEL CONFIG
    # -------------------------------------------------------------------- #
    logger.info("Loading model config...")
    pretrained_checkpoint_path = Path(args['model']['pretrained_checkpoint_path'])
    model_config_file_path = pretrained_checkpoint_path / 'model_config.yaml'
    token_dictionary_file_path = pretrained_checkpoint_path / 'token_dictionary.pkl'
    model_checkpoint_path = pretrained_checkpoint_path / 'model_checkpoint.pt'

    with open(model_config_file_path, 'r') as file:
        model_config = yaml.safe_load(file)

    # -------------------------------------------------------------------- #
    # LOAD TOKEN DICTIONARY
    # -------------------------------------------------------------------- #
    logger.info("Loading token dictionary...")
    n_special_tokens = len(model_config['meta']['special_tokens'])
    seq_len = (
        model_config['data']['seq_len_cell'] +
        model_config['data']['seq_len_neighborhood'] +
        n_special_tokens)

    with open(token_dictionary_file_path, 'rb') as file:
        token_dict = pickle.load(file)
    vocab_size = len(token_dict)
    n_special_values = sum(1 for key in token_dict if "spv" in key)

    # Mirror train.py: if the original training run used protein
    # initialization for the token embedding, the encoder must be
    # rebuilt with the same structure or state_dict loading will fail.
    protein_init_kwargs = parse_protein_init_kwargs(args)
    if protein_init_kwargs is not None:
        protein_init_kwargs['token_dict'] = token_dict

    # -------------------------------------------------------------------- #
    # PREPARE MASK COLLATOR
    # -------------------------------------------------------------------- #
    mask_collator = BlockMaskCollator(
        n_targets=model_config['mask']['n_targets'],
        n_contexts=model_config['mask']['n_contexts'],
        n_segments=model_config['data']['n_segments'],
        seq_len_cell=model_config['data']['seq_len_cell'],
        seq_len_neighborhood=model_config['data']['seq_len_neighborhood'],
        n_special_tokens=n_special_tokens,
        per_block_mask_ratio=model_config['mask']['per_block_mask_ratio'],
        sample_segments=False,
        sample_gene_masks=False
    )

    # -------------------------------------------------------------------- #
    # PREPARE TRAIN DATALOADER
    # -------------------------------------------------------------------- #
    logger.info("Preparing training dataloader...")
    train_cell_dataset = init_cell_dataset(
        dataset=train_dataset,
        vocab_size=vocab_size,
        seq_len_cell=model_config['data']['seq_len_cell'],
        seq_len_neighborhood=model_config['data']['seq_len_neighborhood'],
        tokenizer_type=model_config['data']['tokenizer_type'],
        gt_type=model_config['meta']['gt_type'],
        cell_pos_enc=model_config['meta']['cell_pos_enc'],
        special_tokens=model_config['meta']['special_tokens'],
        sampling_strategy=None,
        n_nonzero_tokens_list=None,
        include_cell_id=True,
        sep_gene_tokens_neb=model_config['data']['sep_gene_tokens_neb'],
        truncate_neighbors=args['data'].get('truncate_neighbors', False),
        tokenized_seq_len_cell=args['data'].get('tokenized_seq_len_cell', None)
    )

    train_loader, _ = init_dataloader_and_sampler(
        cell_dataset=train_cell_dataset,
        batch_size=args['data']['batch_size'],
        distributed=False,
        world_size=1,
        rank=0,
        collate_fn=mask_collator,
        pin_memory=args['data']['pin_memory'],
        num_workers=args['data']['num_workers'],
        drop_last=False,
        persistent_workers=False,
        mega_batch_mult_max=args['data'].get('mega_batch_mult_max', 1000)
    )

    # -------------------------------------------------------------------- #
    # PREPARE VALIDATION DATALOADER (if provided)
    # -------------------------------------------------------------------- #
    val_loader = None
    val_label_lookup = None
    if has_val:
        logger.info("Preparing validation dataloader...")
        val_cell_dataset = init_cell_dataset(
            dataset=val_dataset,
            vocab_size=vocab_size,
            seq_len_cell=model_config['data']['seq_len_cell'],
            seq_len_neighborhood=model_config['data']['seq_len_neighborhood'],
            tokenizer_type=model_config['data']['tokenizer_type'],
            gt_type=model_config['meta']['gt_type'],
            cell_pos_enc=model_config['meta']['cell_pos_enc'],
            special_tokens=model_config['meta']['special_tokens'],
            sampling_strategy=None,
            n_nonzero_tokens_list=None,
            include_cell_id=True,
            sep_gene_tokens_neb=model_config['data']['sep_gene_tokens_neb'],
            truncate_neighbors=args['data'].get('truncate_neighbors', False),
            tokenized_seq_len_cell=args['data'].get('tokenized_seq_len_cell', None)
        )

        val_loader, _ = init_dataloader_and_sampler(
            cell_dataset=val_cell_dataset,
            batch_size=args['data']['batch_size'],
            distributed=False,
            world_size=1,
            rank=0,
            collate_fn=mask_collator,
            pin_memory=args['data']['pin_memory'],
            num_workers=args['data']['num_workers'],
            drop_last=False,
            persistent_workers=False,
            mega_batch_mult_max=args['data'].get('mega_batch_mult_max', 1000)
        )

        # Create validation label lookup
        label_name = args['data']['label_name']
        val_label_lookup = val_adata.obs.set_index('cell_id')[label_name]

    # -------------------------------------------------------------------- #
    # PREPARE EVAL DATALOADER
    # -------------------------------------------------------------------- #
    logger.info(f"Preparing {eval_set_name.lower()} dataloader...")
    eval_cell_dataset = init_cell_dataset(
        dataset=eval_dataset,
        vocab_size=vocab_size,
        seq_len_cell=model_config['data']['seq_len_cell'],
        seq_len_neighborhood=model_config['data']['seq_len_neighborhood'],
        tokenizer_type=model_config['data']['tokenizer_type'],
        gt_type=model_config['meta']['gt_type'],
        cell_pos_enc=model_config['meta']['cell_pos_enc'],
        special_tokens=model_config['meta']['special_tokens'],
        sampling_strategy=None,
        n_nonzero_tokens_list=None,
        include_cell_id=True,
        sep_gene_tokens_neb=model_config['data']['sep_gene_tokens_neb'],
        truncate_neighbors=args['data'].get('truncate_neighbors', False),
        tokenized_seq_len_cell=args['data'].get('tokenized_seq_len_cell', None)
    )

    eval_loader, _ = init_dataloader_and_sampler(
        cell_dataset=eval_cell_dataset,
        batch_size=args['data']['batch_size'],
        distributed=False,
        world_size=1,
        rank=0,
        collate_fn=mask_collator,
        pin_memory=args['data']['pin_memory'],
        num_workers=args['data']['num_workers'],
        drop_last=False,
        persistent_workers=False,
        mega_batch_mult_max=args['data'].get('mega_batch_mult_max', 1000)
    )

    # -------------------------------------------------------------------- #
    # PREPARE LABELS
    # -------------------------------------------------------------------- #
    logger.info("Preparing labels...")
    label_name = args['data']['label_name']

    assert 'cell_id' in train_adata.obs.columns
    assert label_name in train_adata.obs.columns

    label_lookup = train_adata.obs.set_index('cell_id')[label_name]
    eval_label_lookup = eval_adata.obs.set_index('cell_id')[label_name]

    # Set number of classes
    try:
        num_classes = train_adata.uns[f'{label_name}_num_classes']
    except KeyError:
        num_classes = train_adata.obs[label_name].nunique()
        logger.warning(f"Number of classes not found in train_adata.uns. Using nunique() instead.")
    logger.info(f"Number of classes: {num_classes}")

    # -------------------------------------------------------------------- #
    # PREPARE SAVE PATH (optional)
    # -------------------------------------------------------------------- #
    finetune_dir = None
    if save_folder_path:
        finetune_checkpoint_path = Path(save_folder_path)
        current_timestamp = (
                    datetime.now().strftime("%d%m%Y_%H%M%S") +
                    f"_{datetime.now().microsecond // 1000:03d}")
        finetune_dir = finetune_checkpoint_path / current_timestamp
        finetune_dir.mkdir(parents=True, exist_ok=True)

        with open(finetune_dir / 'params.yaml', 'w') as f:
            yaml.dump(args, f)

    # -------------------------------------------------------------------- #
    # LOAD TARGET ENCODER
    # -------------------------------------------------------------------- #
    target_encoder, _ = init_model(
        gt_type=model_config['meta']['gt_type'],
        count_encoding=model_config['meta']['count_encoding'],
        n_value_bins=model_config['meta']['n_value_bins'],
        cell_pos_enc=model_config['meta']['cell_pos_enc'],
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
        mlp_ratio=model_config['meta']['mlp_ratio'],
        use_flash_attention=model_config['meta']['use_flash_attention'],
        api_version=model_config['meta']['api_version'],
        sep_gene_tokens_neb=model_config['data']['sep_gene_tokens_neb'],
        predict_gene=model_config['meta']['predict_gene'],
        pos_learnable=model_config['meta']['pos_learnable'],
        protein_init_kwargs=protein_init_kwargs
    )

    _, _, target_encoder, _, _, _, _ = load_checkpoint(
            device=device,
            r_path=model_checkpoint_path,
            encoder=None,
            predictor=None,
            target_encoder=target_encoder,
            opt=None,
            scaler=None,
            is_training=False
        )

    # -------------------------------------------------------------------- #
    # BUILD MODEL WITH PEFT
    # -------------------------------------------------------------------- #
    use_peft = args['finetune']['use_peft']

    if use_peft:
        peft_method = args['finetune']['peft_method']
        peft_rank = args['finetune']['peft_rank']
        peft_alpha = args['finetune']['peft_alpha']
        peft_dropout = args['finetune']['peft_dropout']
        peft_bias = args['finetune']['peft_bias']
        peft_task_type = args['finetune']['peft_task_type']
        try:
            peft_target_modules = args['finetune']['peft_target_modules']
        except KeyError:
            peft_target_modules = None

        peft_target_encoder = apply_peft(
            target_encoder=target_encoder,
            peft_method=peft_method,
            peft_rank=peft_rank,
            peft_alpha=peft_alpha,
            peft_dropout=peft_dropout,
            peft_bias=peft_bias,
            peft_target_modules=peft_target_modules,
            peft_task_type=peft_task_type
        )
    else:
        # Freeze all parameters except those in target modules
        logger.info("Freezing all parameters except those in target modules...")
        for name, param in target_encoder.named_parameters():
            if any(target in name for target in args['finetune']['peft_target_modules']):
                param.requires_grad = True
                logger.info(f"Trainable: {name}")
            else:
                param.requires_grad = False

        trainable_params = sum(p.numel() for p in target_encoder.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in target_encoder.parameters() if not p.requires_grad)
        logger.info(f"Target encoder: {trainable_params:,} trainable params, {frozen_params:,} frozen params")

    use_mlp = args['finetune']['use_mlp']
    hidden_dim = args['finetune']['hidden_dim']
    lr = args['finetune']['lr']
    num_epochs = args['finetune']['num_epochs']
    patience = args['finetune'].get('patience', 10)
    save_every_k = args['finetune'].get('save_every_k_epochs', -1)
    selection_type = args['finetune']['selection_type']
    agg_excluded_tokens = args['finetune']['excluded_tokens']
    top_k = args['finetune']['top_k']

    model = ClassificationModel(
        base_model=peft_target_encoder if use_peft else target_encoder,
        num_classes=num_classes,
        use_mlp=use_mlp,
        hidden_dim=hidden_dim
    )
    model.to(device)

    # -------------------------------------------------------------------- #
    # PREPARE TRAINING INGREDIENTS
    # -------------------------------------------------------------------- #
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    # Early stopping tracking
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    start_epoch = 0

    # -------------------------------------------------------------------- #
    # RESUME FROM CHECKPOINT (if provided)
    # -------------------------------------------------------------------- #
    if resume_checkpoint_path is not None:
        resume_path = Path(resume_checkpoint_path)
        if resume_path.exists():
            logger.info(f"Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)

            # Load model state
            model.load_state_dict(checkpoint['model'])
            logger.info("Loaded model state from checkpoint")

            # Load optimizer state
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info("Loaded optimizer state from checkpoint")

            # Set start epoch (resume from next epoch)
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                logger.info(f"Resuming from epoch {start_epoch}")

            # Restore best validation loss if available
            if 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
                best_val_loss = checkpoint['val_loss']
                logger.info(f"Restored best val loss: {best_val_loss:.4f}")
        else:
            logger.warning(f"Checkpoint not found at {resume_path}. Starting from scratch.")

    # -------------------------------------------------------------------- #
    # CHECKPOINT SAVING FUNCTION
    # -------------------------------------------------------------------- #
    def save_checkpoint(
            filename,
            epoch,
            train_loss,
            train_accuracy,
            val_loss=None,
            val_accuracy=None
        ):
        if finetune_dir is None:
            return
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'lr': lr,
        }
        checkpoint_path = finetune_dir / f'{filename}.pt'
        torch.save(save_dict, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    # -------------------------------------------------------------------- #
    # TRAINING LOOP
    # -------------------------------------------------------------------- #
    train_loss, train_accuracy = 0.0, 0.0
    val_loss, val_accuracy = None, None

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_running_loss = 0.0
        train_correct_preds = 0
        train_total_preds = 0

        for itr, (udata, _, _, masks_attention, _) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}"):
            for key in udata.keys():
                if key != 'cell_id':
                    udata[key] = udata[key].to(device, non_blocking=True)
            masks_attention = masks_attention.to(device, non_blocking=True)

            ns_tokens = udata['tokens'][:, n_special_tokens:]
            # Create cell mask or neighborhood mask for aggregation depending on the selection type (agg_cell or agg_graph)
            if selection_type == 'agg_cell':
                selection_mask = create_binary_selection_mask(
                    ns_tokens,
                    selection_type=selection_type,
                    excluded_tokens=agg_excluded_tokens,
                    seq_len_cell=model_config['data']['seq_len_cell'],
                    top_k=top_k,
                )
            elif selection_type == 'agg_graph':
                selection_mask = create_binary_selection_mask(
                    ns_tokens,
                    selection_type=selection_type,
                    excluded_tokens=agg_excluded_tokens,
                    seq_len_cell=model_config['data']['seq_len_cell'],
                    top_k=top_k,
                    n_segments=model_config['data']['n_segments']
                )
            else:
                raise ValueError(f"Selection type {selection_type} not supported.")

            optimizer.zero_grad()
            logits = model(
                udata=udata,
                masks_attention=masks_attention,
                selection_mask=selection_mask,
            )

            labels = torch.tensor(
                label_lookup[udata['cell_id']].values,
                dtype=torch.long,
                device=device
            )

            train_loss_batch = criterion(input=logits, target=labels)
            train_loss_batch.backward()
            optimizer.step()

            train_running_loss += train_loss_batch.item()

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                train_correct_preds += (preds == labels).sum().item()
                train_total_preds += labels.size(0)

            # if itr > 5:
                # break  # For debugging, remove this line for full training

        train_loss = train_running_loss / len(train_loader)
        train_accuracy = train_correct_preds / train_total_preds

        # Early stopping on validation (if provided)
        if has_val:
            val_loss, val_accuracy = validate(
                model=model,
                val_loader=val_loader,
                val_label_lookup=val_label_lookup,
                criterion=criterion,
                device=device,
                n_special_tokens=n_special_tokens,
                model_config=model_config,
                selection_type=selection_type,
                agg_excluded_tokens=agg_excluded_tokens,
                top_k=top_k
            )

            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                save_checkpoint('checkpoint_best', epoch, train_loss, train_accuracy, val_loss, val_accuracy)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            if save_every_k > 0 and (epoch + 1) % save_every_k == 0:
                save_checkpoint(f'checkpoint_epoch_{epoch+1}', epoch, train_loss, train_accuracy, val_loss, val_accuracy)
        else:
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
            )

            if save_every_k > 0 and (epoch + 1) % save_every_k == 0:
                save_checkpoint(f'checkpoint_epoch_{epoch+1}', epoch, train_loss, train_accuracy)

    save_checkpoint(
        'checkpoint_last',
        epoch,
        train_loss,
        train_accuracy,
        val_loss,
        val_accuracy
    )

    # Load best model if we did early stopping
    if has_val and best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    # -------------------------------------------------------------------- #
    # FINAL EVALUATION
    # -------------------------------------------------------------------- #
    logger.info(f"Evaluating on {eval_set_name} set...")
    predictions, targets, accuracy = test(
        model=model,
        loader=eval_loader,
        label_lookup=eval_label_lookup,
        device=device,
        n_special_tokens=n_special_tokens,
        model_config=model_config,
        selection_type=selection_type,
        agg_excluded_tokens=agg_excluded_tokens,
        top_k=top_k
    )

    logger.info(f"{eval_set_name} Accuracy: {accuracy:.4f}")

    return predictions, targets, accuracy


if __name__ == '__main__':

    args = parse_arguments()

    cols = None

    train_dataset = load_from_disk(args['data']['train_dataset'])
    cols = train_dataset.column_names
    train_dataset.set_format(
        type="torch",
        columns=cols,
        output_all_columns=True
    )
    train_adata = sc.read_h5ad(args['data']['train_adata'])

    val_dataset, val_adata = None, None
    if 'val_dataset' in args['data'] and 'val_adata' in args['data']:
        val_dataset = load_from_disk(args['data']['val_dataset'])
        val_dataset.set_format(
            type="torch",
            columns=cols,
            output_all_columns=True
        )
        val_adata = sc.read_h5ad(args['data']['val_adata'])

    test_dataset, test_adata = None, None
    if 'test_dataset' in args['data'] and 'test_adata' in args['data']:
        test_dataset = load_from_disk(args['data']['test_dataset'])
        test_dataset.set_format(
            type="torch",
            columns=cols,
            output_all_columns=True
        )
        test_adata = sc.read_h5ad(args['data']['test_adata'])

    save_folder_path = (
        args['model'].get('save_folder_path') or
        args['model'].get('finetune_checkpoint_path')
    )

    finetune(
        args=args,
        train_adata=train_adata,
        train_dataset=train_dataset,
        val_adata=val_adata,
        val_dataset=val_dataset,
        test_adata=test_adata,
        test_dataset=test_dataset,
        save_folder_path=save_folder_path,
    )
