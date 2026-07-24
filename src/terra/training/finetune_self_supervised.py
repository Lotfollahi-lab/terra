"""Self-supervised (I-JEPA) LoRA fine-tuning of a pretrained TERRA model.

This adapts a pretrained TERRA encoder to a new data domain (e.g. adapting a
Xenium-pretrained model to 10x Visium) by continuing the **self-supervised
I-JEPA objective** on the new data while training only low-rank (LoRA)
adapters. It is the counterpart to :func:`terra.training.finetune.finetune`,
which instead fine-tunes with a **supervised** classification head.

The pretrained weights are frozen; a context encoder + predictor learn to match
the representation of masked target blocks produced by an EMA target encoder.
Only the LoRA adapters on the attention/MLP projections are trained, and the
target encoder is updated by EMA of the (adapter-merged) context encoder.

Checkpoints are written as ``checkpoint_epoch_<N>.pt`` holding the ``encoder``,
``predictor`` and ``target_encoder`` state dicts. Use
:func:`prepare_finetuned_model` (also in this module) to merge the LoRA adapters
and assemble a self-contained, embeddable model folder.
"""

import copy
import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset, concatenate_datasets, load_from_disk
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from terra.datasets.cell_datasets import init_cell_dataset
from terra.datasets.dataloaders import init_dataloader_and_sampler
from terra.masks.block_masking import BlockMaskCollator
from terra.masks.utils import apply_masks
from terra.utils.helper import init_model, init_opt

logger = logging.getLogger(__name__)

_GLOBAL_SEED = 0


def _apply_lora(model,
                peft_rank=16,
                peft_alpha=256,
                peft_dropout=0.1,
                peft_bias='none',
                peft_target_modules=None):
    """Wrap ``model`` with LoRA adapters and freeze everything else."""
    if peft_target_modules is None:
        peft_target_modules = ['qkv', 'proj', 'fc1', 'fc2']
    peft_config = LoraConfig(r=peft_rank,
                             lora_alpha=peft_alpha,
                             lora_dropout=peft_dropout,
                             bias=peft_bias,
                             task_type=None,
                             target_modules=peft_target_modules)
    peft_model = get_peft_model(model, peft_config)
    for name, p in peft_model.named_parameters():   # train only the adapters
        p.requires_grad = ('lora_' in name.lower())
    return peft_model


def _load_pretrained_checkpoint(device, r_path, encoder, predictor,
                                target_encoder):
    """Load pretrained weights, stripping any ``module.`` DDP prefix."""
    ckpt = torch.load(r_path, map_location=torch.device(device))
    _strip = lambda sd: {k.replace('module.', ''): v for k, v in sd.items()}
    if encoder is not None:
        logger.info('encoder load: %s',
                    encoder.load_state_dict(_strip(ckpt['encoder'])))
    if predictor is not None:
        logger.info('predictor load: %s',
                    predictor.load_state_dict(_strip(ckpt['predictor'])))
    if target_encoder is not None:
        logger.info('target_encoder load: %s',
                    target_encoder.load_state_dict(
                        _strip(ckpt['target_encoder'])))
    del ckpt
    return encoder, predictor, target_encoder


def _load_finetune_dataset(dataset_paths: list[str]) -> Dataset:
    """Load, annotate, concatenate and shuffle tokenised section datasets."""
    logger.info("Loading %d dataset(s)...", len(dataset_paths))
    datasets = []
    for i, path in enumerate(dataset_paths):
        logger.info("Loading dataset %d/%d: %s", i + 1, len(dataset_paths), path)
        ds = load_from_disk(path)
        # Annotate each source file with a 'section' column for group splits
        ds = ds.add_column('section', [Path(path).name] * len(ds))
        datasets.append(ds)
    dataset = concatenate_datasets(datasets)
    dataset = dataset.shuffle(seed=_GLOBAL_SEED)
    # Set torch format, excluding non-tensorable metadata columns
    cols = [c for c in dataset.column_names if c != 'section']
    dataset.set_format(type="torch", columns=cols, output_all_columns=True)
    logger.info("Loaded %d samples", len(dataset))
    return dataset


def finetune_self_supervised(
        args: dict,
        dataset: Dataset | None = None,
        save_folder_path: str | None = None,
        run_name: str | None = None,
        use_wandb: bool = False,
        LOCAL_RANK: int | None = None,
        WORLD_RANK: int | None = None,
    ) -> str:
    """Self-supervised (I-JEPA) LoRA fine-tuning of a pretrained TERRA model.

    Parameters
    ----------
    args:
        Configuration dictionary with ``model``, ``data`` and ``finetune``
        sections (see ``configs`` / the tutorial's ``config_Visium_IJEPA.yaml``
        for the schema). ``args['model']['pretrained_checkpoint_path']`` must
        point at a model folder containing ``model_config.yaml``,
        ``token_dictionary.pkl`` and ``model_checkpoint.pt``.
    dataset:
        Tokenised HuggingFace :class:`~datasets.Dataset` to fine-tune on. If
        ``None``, the tokenised sections listed in
        ``args['data']['finetune_dataset']`` are loaded, concatenated and
        shuffled automatically.
    save_folder_path:
        Directory for checkpoints. Defaults to
        ``args['model']['finetune_checkpoint_path']``.
    run_name:
        Name of the run subdirectory. Defaults to a timestamp.
    use_wandb:
        If ``True``, log training metrics to Weights & Biases. A run is created
        if one is not already active, and finished at the end.
    LOCAL_RANK, WORLD_RANK:
        Optional distributed ranks (single-process by default).

    Returns
    -------
    str
        Path to the run directory where checkpoints were written.
    """
    # -- Optional Weights & Biases logging ------------------------------------
    _wandb = None
    _owns_wandb = False
    if use_wandb:
        import wandb as _wandb
        if _wandb.run is None:
            _wandb.init(
                project=args.get('wandb', {}).get('project', 'terra-finetune'),
                name=run_name,
                config=args)
            _owns_wandb = True

    # -- Backend setup --------------------------------------------------------
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_GLOBAL_SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    elif LOCAL_RANK is not None:
        device = torch.device(f"cuda:{LOCAL_RANK}")
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- Load the tokenised dataset if not provided ---------------------------
    if dataset is None:
        dataset = _load_finetune_dataset(args['data']['finetune_dataset'])

    # -- Load pretrained model paths + config ---------------------------------
    pretrained_checkpoint_path = Path(args['model']['pretrained_checkpoint_path'])
    model_config_file_path = pretrained_checkpoint_path / 'model_config.yaml'
    token_dictionary_file_path = pretrained_checkpoint_path / 'token_dictionary.pkl'
    model_checkpoint_path = pretrained_checkpoint_path / 'model_checkpoint.pt'

    with open(model_config_file_path, 'r') as file:
        model_config = yaml.safe_load(file)

    n_special_tokens = len(model_config['meta']['special_tokens'])
    seq_len = (
        model_config['data']['seq_len_cell'] +
        model_config['data']['seq_len_neighborhood'] +
        n_special_tokens)

    with open(token_dictionary_file_path, 'rb') as file:
        token_dict = pickle.load(file)
    vocab_size = len(token_dict)
    n_special_values = sum(1 for key in token_dict if "spv" in key)

    # -- Mask generator + tokenised dataset + dataloader ----------------------
    mask_collator = BlockMaskCollator(
        n_targets=model_config['mask']['n_targets'],
        n_contexts=model_config['mask']['n_contexts'],
        n_segments=model_config['data']['n_segments'],
        seq_len_cell=model_config['data']['seq_len_cell'],
        seq_len_neighborhood=model_config['data']['seq_len_neighborhood'],
        n_special_tokens=n_special_tokens,
        per_block_mask_ratio=model_config['mask']['per_block_mask_ratio'],
        sample_segments=args['data'].get('sample_segments', False),
        sample_gene_masks=args['data'].get('sample_gene_masks', False),
    )

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
        n_nonzero_tokens_list=None,
        include_cell_id=True,
        sep_gene_tokens_neb=model_config['data']['sep_gene_tokens_neb'])

    loader, _ = init_dataloader_and_sampler(
        cell_dataset=cell_dataset,
        batch_size=args['data']['batch_size'],
        distributed=False,
        world_size=1,
        rank=0,
        collate_fn=mask_collator,
        pin_memory=args['data']['pin_memory'],
        num_workers=args['data']['num_workers'],
        drop_last=args['data'].get('drop_last', True),
        persistent_workers=False,
        mega_batch_mult_max=1000)

    # -- Checkpoint directory -------------------------------------------------
    if not save_folder_path:
        finetune_checkpoint_path = Path(args['model']['finetune_checkpoint_path'])
    else:
        finetune_checkpoint_path = Path(save_folder_path)

    if run_name:
        finetune_dir = finetune_checkpoint_path / run_name
    else:
        current_timestamp = (
            datetime.now().strftime("%d%m%Y_%H%M%S") +
            f"_{datetime.now().microsecond // 1000:03d}")
        finetune_dir = finetune_checkpoint_path / current_timestamp
    finetune_dir.mkdir(parents=True, exist_ok=True)

    with open(finetune_dir / 'params.yaml', 'w') as f:
        yaml.dump(args, f)

    # -- Initialize encoder, predictor and target encoder ---------------------
    logger.info("Initializing encoder, predictor, and target encoder...")

    def _build_model():
        return init_model(
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
            mlp_bias=model_config['meta'].get('mlp_bias', True))

    encoder, predictor = _build_model()
    target_encoder, _ = _build_model()

    logger.info("Loading pretrained weights from %s...", model_checkpoint_path)
    encoder, predictor, target_encoder = _load_pretrained_checkpoint(
        device=device,
        r_path=model_checkpoint_path,
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder)

    # Freeze the target encoder
    for p in target_encoder.parameters():
        p.requires_grad = False
    target_encoder.eval()

    # -- Low-rank adaptation (LoRA) -------------------------------------------
    use_peft = args['finetune'].get('use_peft', False)
    if use_peft:
        logger.info("Applying LoRA adapters")
        peft_target_modules = args['finetune'].get(
            'peft_target_modules', ['qkv', 'proj', 'fc1', 'fc2'])
        encoder = _apply_lora(
            model=encoder,
            peft_rank=args['finetune']['peft_rank'],
            peft_alpha=args['finetune']['peft_alpha'],
            peft_dropout=args['finetune']['peft_dropout'],
            peft_bias=args['finetune']['peft_bias'],
            peft_target_modules=peft_target_modules)
        predictor = _apply_lora(
            model=predictor,
            peft_rank=args['finetune']['peft_rank'],
            peft_alpha=args['finetune']['peft_alpha'],
            peft_dropout=args['finetune']['peft_dropout'],
            peft_bias=args['finetune']['peft_bias'],
            peft_target_modules=peft_target_modules)

        # Freeze base params, keep LoRA params trainable
        for model in (encoder, predictor):
            for name, param in model.named_parameters():
                param.requires_grad = ('lora_' in name.lower())

        n_trainable_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        n_trainable_pred = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
        logger.info("Trainable params - encoder: %d, predictor: %d",
                    n_trainable_enc, n_trainable_pred)

        # Assert base params are frozen
        for model in (encoder, predictor):
            for name, p in model.named_parameters():
                if 'lora_' not in name.lower():
                    assert p.requires_grad is False, f"Parameter {name} should be frozen"

    encoder.to(device)
    predictor.to(device)
    target_encoder.to(device)

    # -- Optimizer, schedulers, EMA -------------------------------------------
    lr = args['finetune']['lr']
    start_lr = args['finetune'].get('start_lr', lr)
    final_lr = args['finetune'].get('final_lr', 1.0e-06)
    num_epochs = args['finetune']['num_epochs']
    warmup_epochs = args['finetune'].get('warmup_epochs', 0)
    weight_decay = args['finetune'].get('weight_decay', 0.04)
    final_weight_decay = args['finetune'].get('final_weight_decay', 0.4)
    ema_momentum = args['finetune'].get('ema_momentum', 0.996)
    final_ema_momentum = args['finetune'].get('final_ema_momentum', 1.0)
    loss_fn_type = args['finetune'].get('loss_fn_type', 'smooth_l1')
    clip_grad = args['finetune'].get('clip_grad', None)
    use_bfloat16 = args['finetune'].get('use_bfloat16', False)

    optimizer, scaler, lr_schedule, wd_schedule = init_opt(
        encoder=encoder,
        predictor=predictor,
        iterations_per_epoch=len(loader),
        start_lr=start_lr,
        ref_lr=lr,
        warmup=warmup_epochs,
        num_epochs=num_epochs,
        wd=weight_decay,
        final_wd=final_weight_decay,
        final_lr=final_lr,
        use_bfloat16=use_bfloat16)

    opt_trainable_params = sum(
        p.numel() for g in optimizer.param_groups
        for p in g['params'] if p.requires_grad)
    logger.info("Optimizer trainable parameters: %d", opt_trainable_params)

    ipe = len(loader)
    ema = [ema_momentum, final_ema_momentum]
    mom_schedule = (ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs)
                    for i in range(ipe * num_epochs))
    momentum_scheduler_iter = iter(mom_schedule)

    epoch_loss = 0.0

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': epoch_loss,
            'batch_size': args['data']['batch_size'],
            'world_size': 1,
            'lr': lr,
        }
        save_path = finetune_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(save_dict, save_path)
        logger.info("Checkpoint saved to %s", save_path)

    def call_encoder(model, batch, masks=None, masks_attention=None):
        """Call encoder, handling the PEFT wrapper if present."""
        if use_peft:
            return model.base_model.model(
                batch=batch, masks=masks, masks_attention=masks_attention)
        return model(batch=batch, masks=masks, masks_attention=masks_attention)

    def call_predictor(model, z, token_emb, batch, masks_enc, masks_pred,
                       masks_attention=None):
        """Call predictor, handling the PEFT wrapper if present."""
        if use_peft:
            return model.base_model.model(
                z=z, token_emb=token_emb, batch=batch,
                masks_enc=masks_enc, masks_pred=masks_pred,
                masks_attention=masks_attention)
        return model(
            z=z, token_emb=token_emb, batch=batch,
            masks_enc=masks_enc, masks_pred=masks_pred,
            masks_attention=masks_attention)

    # -- Training loop --------------------------------------------------------
    logger.info("Starting I-JEPA self-supervised fine-tuning...")
    logger.info("Start LR: %s, Main LR: %s, Final LR: %s, Warmup epochs: %s",
                start_lr, lr, final_lr, warmup_epochs)
    logger.info("Weight decay: %s -> %s", weight_decay, final_weight_decay)
    logger.info("EMA momentum: %s -> %s", ema_momentum, final_ema_momentum)

    for epoch in range(num_epochs):
        logger.info("Epoch %d/%d", epoch + 1, num_epochs)
        running_loss = 0.0
        log_grad_stats = True
        encoder.train()
        predictor.train()

        for itr, (udata, masks_enc, masks_pred, masks_attention) in enumerate(tqdm(loader)):
            for key, val in udata.items():
                if isinstance(val, torch.Tensor):
                    udata[key] = val.to(device, non_blocking=True)
            masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
            masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]
            masks_attention = masks_attention.to(device, non_blocking=True)

            _new_lr = lr_schedule.step()
            _new_wd = wd_schedule.step()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
                # Target encoder forward pass
                with torch.no_grad():
                    h, _ = target_encoder(batch=udata, masks_attention=masks_attention)
                    h = F.layer_norm(h, (h.size(-1),))
                    h = apply_masks(h, masks_pred, concat=False)

                # Context encoder forward pass
                z, token_emb = call_encoder(
                    encoder, batch=udata, masks=masks_enc, masks_attention=None)

                # Predictor forward pass
                z = call_predictor(
                    predictor, z=z, token_emb=token_emb, batch=udata,
                    masks_enc=masks_enc, masks_pred=masks_pred, masks_attention=None)

                # Loss
                loss = 0.
                for zi, hi in zip(z, h):
                    if loss_fn_type == 'smooth_l1':
                        loss += F.smooth_l1_loss(zi, hi)
                    elif loss_fn_type == 'l1':
                        loss += torch.mean(torch.abs(zi - hi))
                    elif loss_fn_type == 'mse':
                        loss += F.mse_loss(zi, hi)
                loss /= len(masks_pred)

            # Backward pass
            n_trainable = (
                sum(p.numel() for p in encoder.parameters() if p.requires_grad) +
                sum(p.numel() for p in predictor.parameters() if p.requires_grad))
            if n_trainable == 0:
                raise RuntimeError(
                    "No trainable parameters found in encoder or predictor before "
                    "backward(). This likely indicates PEFT adapters were "
                    "inadvertently frozen.")

            loss.backward()

            if clip_grad is not None:
                grad_norm_enc = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                grad_norm_pred = torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                if log_grad_stats:
                    logger.info("Epoch %d, Batch %d: Grad norm - Encoder: %.4f, "
                                "Predictor: %.4f", epoch + 1, itr + 1,
                                grad_norm_enc, grad_norm_pred)
                    logger.info("Current LR: %.6f, WD: %.6f", _new_lr, _new_wd)
                    log_grad_stats = False

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # EMA update of the target encoder
            with torch.no_grad():
                m = next(momentum_scheduler_iter)
                if use_peft:
                    # Merge adapters into a temporary copy, then EMA-update
                    temp_encoder = copy.deepcopy(encoder)
                    merged = temp_encoder.merge_and_unload()
                    enc_state_dict = merged.state_dict()
                    device_target = next(target_encoder.parameters()).device
                    for name, tensor in enc_state_dict.items():
                        if isinstance(tensor, torch.Tensor):
                            enc_state_dict[name] = tensor.to(device_target)
                    target_state_dict = target_encoder.state_dict()
                    for name, t in target_state_dict.items():
                        if name in enc_state_dict:
                            t.mul_(m).add_((1. - m) * enc_state_dict[name].detach())
                    target_encoder.load_state_dict(target_state_dict)
                    del temp_encoder, merged, enc_state_dict
                else:
                    for param_q, param_k in zip(encoder.parameters(),
                                                target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)

            running_loss += loss.item()

            if use_wandb and (itr + 1) % 10 == 0:
                _wandb.log({
                    "batch_loss": loss.item(),
                    "avg_loss": running_loss / (itr + 1),
                    "batch": epoch * len(loader) + itr,
                    "learning_rate_batch": _new_lr,
                    "weight_decay_batch": _new_wd,
                })

        epoch_loss = running_loss / len(loader)
        logger.info("Epoch [%d/%d], Loss: %.4f, LR: %.6f",
                    epoch + 1, num_epochs, epoch_loss, _new_lr)
        if use_wandb:
            _wandb.log({"epoch": epoch + 1, "loss": epoch_loss,
                        "learning_rate": _new_lr})

        if (epoch + 1) % args['finetune'].get('save_every', 10) == 0:
            save_checkpoint(epoch + 1)

    save_checkpoint(num_epochs)
    logger.info("Fine-tuning complete!")

    if _owns_wandb:
        _wandb.finish()

    return str(finetune_dir)



# Adapter merge / model preparation

def merge_lora_weights(state_dict: dict) -> dict:
    """Fold LoRA adapters in a PEFT state dict into the base weights.

    For each adapted layer, ``W = W_base + (lora_B @ lora_A)``; layers without
    adapters are copied unchanged. The ``base_model.model.`` / ``base_layer``
    PEFT wrapper prefixes are stripped so the result matches the plain model's
    parameter names.
    """
    base_weights, lora_a_weights, lora_b_weights = {}, {}, {}
    for key, value in state_dict.items():
        if 'base_layer.weight' in key or 'base_layer.bias' in key:
            clean_key = key.replace('base_model.model.', '').replace('.base_layer', '')
            base_weights[clean_key] = value
        elif 'lora_A.default.weight' in key:
            clean_key = key.replace('base_model.model.', '').replace('.lora_A.default.weight', '.weight')
            lora_a_weights[clean_key] = value
        elif 'lora_B.default.weight' in key:
            clean_key = key.replace('base_model.model.', '').replace('.lora_B.default.weight', '.weight')
            lora_b_weights[clean_key] = value
        elif 'base_model.model.' in key:
            clean_key = key.replace('base_model.model.', '')
            base_weights[clean_key] = value

    merged_weights = {}
    n_merged = 0
    for key, base_weight in base_weights.items():
        if key in lora_a_weights and key in lora_b_weights:
            delta = torch.mm(lora_b_weights[key], lora_a_weights[key])
            merged_weights[key] = base_weight + delta
            n_merged += 1
        else:
            merged_weights[key] = base_weight
    logger.info("Merged %d LoRA adapter layers (%d total layers)",
                n_merged, len(merged_weights))
    return merged_weights


def prepare_finetuned_model(
        finetuned_checkpoint_dir: str,
        pretrained_model_dir: str,
        output_dir: str,
        checkpoint_epoch: int | None = None,
        use_peft: bool = False,
    ) -> str:
    """Assemble an embeddable model folder from a fine-tuning checkpoint.

    Copies ``model_config.yaml`` and ``token_dictionary.pkl`` from the
    pretrained model, loads a checkpoint written by
    :func:`finetune_self_supervised`, optionally merges its LoRA adapters into
    the base weights, and writes a self-contained model folder
    (``model_config.yaml``, ``token_dictionary.pkl``, ``model_checkpoint.pt``,
    ``finetuning_metadata.yaml``) that :func:`terra.tokenize_adata` /
    :func:`terra.embed_dataset` can consume directly.

    Parameters
    ----------
    finetuned_checkpoint_dir:
        Run directory containing ``checkpoint_epoch_<N>.pt`` files (the value
        returned by :func:`finetune_self_supervised`).
    pretrained_model_dir:
        The pretrained model folder the fine-tune started from (source of the
        config and token dictionary).
    output_dir:
        Destination folder for the prepared, embeddable model.
    checkpoint_epoch:
        Epoch checkpoint to prepare. Defaults to the highest-epoch checkpoint.
    use_peft:
        Set ``True`` if the fine-tune used LoRA, to merge the adapters into the
        base encoder/predictor weights. The target encoder (used for embedding)
        already holds fully EMA-merged weights and is copied as-is.

    Returns
    -------
    str
        Path to the prepared model folder (``output_dir``).
    """
    finetuned_checkpoint_dir = Path(finetuned_checkpoint_dir)
    pretrained_model_dir = Path(pretrained_model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config + vocabulary come from the pretrained model unchanged
    shutil.copy2(pretrained_model_dir / 'model_config.yaml',
                 output_dir / 'model_config.yaml')
    shutil.copy2(pretrained_model_dir / 'token_dictionary.pkl',
                 output_dir / 'token_dictionary.pkl')

    # Locate the checkpoint to prepare
    if checkpoint_epoch is not None:
        checkpoint_file = finetuned_checkpoint_dir / f'checkpoint_epoch_{checkpoint_epoch}.pt'
    else:
        checkpoints = list(finetuned_checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found in {finetuned_checkpoint_dir}")
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        checkpoint_file = checkpoints[-1]
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    logger.info("Loading finetuned checkpoint: %s", checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location='cpu')

    encoder_state_dict = checkpoint['encoder']
    predictor_state_dict = checkpoint['predictor']
    target_encoder_state_dict = checkpoint['target_encoder']

    if use_peft:
        logger.info("Merging LoRA adapters into the base model...")
        encoder_state_dict = merge_lora_weights(encoder_state_dict)
        predictor_state_dict = merge_lora_weights(predictor_state_dict)
        # target_encoder already holds merged fine-tuned weights from the EMA

    model_checkpoint = {
        'encoder': encoder_state_dict,
        'predictor': predictor_state_dict,
        'target_encoder': target_encoder_state_dict,
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', 0.0),
    }
    torch.save(model_checkpoint, output_dir / 'model_checkpoint.pt')

    metadata = {
        'source_checkpoint': str(checkpoint_file),
        'pretrained_model': str(pretrained_model_dir),
        'finetuned_epoch': checkpoint.get('epoch', 'unknown'),
        'final_loss': float(checkpoint.get('loss', 0.0)),
        'use_peft': use_peft,
    }
    with open(output_dir / 'finetuning_metadata.yaml', 'w') as f:
        yaml.dump(metadata, f)

    logger.info("Prepared embeddable model at %s", output_dir)
    return str(output_dir)
