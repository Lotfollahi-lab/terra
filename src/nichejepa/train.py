"""
Adapted from Assran, M. et al. Self-supervised learning from images with a 
Joint-Embedding Predictive Architecture. Proc. IEEE Comput. Soc. Conf. Comput.
Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/train.py (05.06.2024).
"""

import os
import pickle

"""
# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass
"""

import copy
import logging
import sys
import yaml
from datetime import datetime

import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from nichejepa.datasets.cell_datasets import make_cell_dataset
from nichejepa.datasets.dataloaders import init_dataloader_and_sampler
from .helper import init_model, init_opt, load_checkpoint
from nichejepa.masks.random_masking import RandomMaskCollator
from nichejepa.masks.block_masking  import BlockMaskCollator
from nichejepa.masks.utils import apply_masks
from nichejepa.models.utils import repeat_interleave_batch
from nichejepa.utils.distributed import init_distributed
from nichejepa.utils.logging import (AverageMeter,
                                     CSVLogger,
                                     grad_logger)

# os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1" # Better error propagation

_GLOBAL_SEED = 0


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def train(args: dict,
          train_dataset: datasets.Dataset,
          resume_preempt: bool = False,
          save_folder_path: str | None = None,
          LOCAL_RANK: int | None = None,
          WORLD_RANK: int | None = None,
          ):
    """
    Train model.

    Parameters
    -----------
    args:
        Dictionary containing the hyperparams from the config file.
    train_dataset:
        Train split of the huggingface dataset.
    resume_preempt:
        If `True`, resume a preempted job.
    save_folder_path:
        Path for saving model artifacts.
    LOCAL_RANK:
        Local rank of the process.
    WORLD_RANK:
        World rank of the process.
    """
    # Set random seeds
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_GLOBAL_SEED)
    torch.backends.cudnn.deterministic = False # set to True for reproducibility
    torch.backends.cudnn.benchmark = True # set to False for reproducibility

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    elif LOCAL_RANK is not None:
        device = torch.device(f"cuda:{LOCAL_RANK}")
    elif LOCAL_RANK is None:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # Load params from config file
    dataset_name = args['data']['dataset_name']
    tokenizer_type = args['data']['tokenizer_type']
    n_special_values = args['data']['n_special_values']
    vocab_size = args['data']['vocab_size']
    seq_len_cell = args['data']['seq_len_cell']
    seq_len_neighborhood = args['data']['seq_len_neighborhood']
    n_segments = args['data']['n_segments']
    sampling_strategy = args['data']['sampling_strategy']
    batch_size = args['data']['batch_size']
    num_workers = args['data']['num_workers']
    pin_memory = args['data']['pin_memory']
    if 'precomputed_n_nonzero_tokens' in args['data'].keys():
        if args['data']['precomputed_n_nonzero_tokens']:
            with open(args['data']['precomputed_n_nonzero_tokens'], "rb") as f: 
                n_nonzero_tokens = pickle.load(f)
        else:
            n_nonzero_tokens = None
    else:
        n_nonzero_tokens = None

    gt_type = args['meta']['gt_type']
    enc_depth = args['meta']['enc_depth'] 
    enc_emb_dim = args['meta']['enc_emb_dim']    
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    special_tokens = args['meta']['special_tokens']
    pos_learnable = args['meta']['pos_learnable']
    seg_learnable = args['meta']['seg_learnable']
    use_bfloat16 = args['meta']['use_bfloat16']
    if 'new_spc' in args['meta'].keys():
        new_spc = args['meta']['new_spc']
    else:
        new_spc = False
    if 'loss_fn_type' in args['meta'].keys():
        loss_fn_type = args['meta']['loss_fn_type']
    else:
        loss_fn_type = 'l1'

    n_contexts = args['mask']['n_contexts']
    n_targets = args['mask']['n_targets']
    block_masking = args['mask']['block_masking']
    context_mask_size = args['mask']['context_mask_size']
    target_mask_size = args['mask']['target_mask_size']
    per_block_mask_ratio = args['mask']['per_block_mask_ratio']
    if 'restrict_special_attention' in args['meta'].keys():
        restrict_special_attention = args['meta']['restrict_special_attention']
    else:
        restrict_special_attention = False
    if 'sample_segments' in args['meta'].keys():
        sample_segments = args['meta']['sample_segments']
    else:
        sample_segments = False

    warmup = args['optimization']['warmup']
    num_epochs = args['optimization']['epochs']
    if isinstance(args['optimization']['ema'], list):
       ema = args['optimization']['ema']
    else:
       ema = [args['optimization']['ema'], 1]
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    ipe_scale = args['optimization']['ipe_scale'] # scheduler scale factor
    clip_grad = args['optimization']['clip_grad']

    log_freq = args['state']['log_freq']
    checkpoint_freq = args['state']['checkpoint_freq']
    write_tag = args['state']['write_tag']
    load_model = args['state']['load_checkpoint'] or resume_preempt
    r_file = args['state']['read_checkpoint']

    # Define tokenizer-specific params
    if tokenizer_type == 'cell_neighborhood':
        max_special_tokens = 7
        max_cls_tokens = args['meta']['n_cls']
        special_tokens = ['cls_cell', 'cls_neighborhood'] + special_tokens
    elif tokenizer_type == 'cell_graph':
        max_special_tokens = 105
        max_cls_tokens = args['meta']['n_cls']
        special_tokens = [
            f'cls_{i}' for i in range(max_cls_tokens)] + special_tokens

    # Get token sequence length and number of special tokens
    n_special_tokens = len(special_tokens)
    seq_len = seq_len_cell + seq_len_neighborhood + n_special_tokens

    # Initialize torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}.')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # Create folder to store artifacts
    if not save_folder_path:
        artifact_folder_path = os.path.join(
            os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))), "artifacts")
        current_timestamp = (
            datetime.now().strftime("%d%m%Y_%H%M%S") +
            f"_{datetime.now().microsecond // 1000:03d}")
        save_folder_path = os.path.join(artifact_folder_path,
                                        dataset_name,
                                        current_timestamp)
    if rank==0:
        os.makedirs(save_folder_path, exist_ok=True)

    # Store config file with model
    if rank==0:
        dump = os.path.join(save_folder_path, 'params.yaml')
        with open(dump, 'w') as f:
            yaml.dump(args, f)

    # Define log/checkpointing paths
    log_file = os.path.join(save_folder_path, f'{write_tag}_r{rank}.csv')
    save_path = os.path.join(save_folder_path,
                             f'{write_tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(save_folder_path, f'{write_tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(
            load_folder_path, r_file) if r_file is not None else latest_path


    # Initialize encoder, predictor and target encoder
    encoder, predictor = init_model(
        gt_type=gt_type,
        device=device,
        vocab_size=vocab_size,
        seq_len=seq_len,
        max_cls_tokens=max_cls_tokens,
        max_special_tokens=max_special_tokens,
        n_special_tokens=n_special_tokens,
        n_segments=n_segments,
        n_special_values=n_special_values,
        enc_emb_dim=enc_emb_dim,
        enc_depth=enc_depth,
        pred_emb_dim=pred_emb_dim,
        pred_depth=pred_depth,
        pos_learnable=pos_learnable,
        seg_learnable=seg_learnable,
        new_spc=new_spc)
    target_encoder = copy.deepcopy(encoder)

    # Initialize mask collator
    if block_masking:
       mask_collator = BlockMaskCollator(
            n_targets=n_targets,
            n_contexts=n_contexts,
            seq_len_cell=seq_len_cell,
            seq_len_neighborhood=seq_len_neighborhood,
            max_special_tokens=max_special_tokens,
            n_special_tokens=n_special_tokens,
            max_cls_tokens=max_cls_tokens,
            per_block_mask_ratio=per_block_mask_ratio,
            restrict_special_attention=restrict_special_attention,
            sample_segments=sample_segments)
    else:
        mask_collator = RandomMaskCollator(
            n_targets=n_targets,
            n_contexts=n_contexts,
            seq_len_cell=seq_len_cell,
            seq_len_neighborhood=seq_len_neighborhood,
            n_special_tokens=n_special_tokens,
            target_mask_size=target_mask_size,
            context_mask_size=context_mask_size,)
    
    # Initialize train and test datasets, dataloaders and samplers
    train_cell_dataset = make_cell_dataset(
        dataset=train_dataset,
        vocab_size=vocab_size,
        seq_len_cell=seq_len_cell,
        seq_len_neighborhood=seq_len_neighborhood,
        max_cls_tokens=max_cls_tokens,
        max_special_tokens=max_special_tokens,
        tokenizer_type=tokenizer_type,
        gt_type=gt_type,
        special_tokens=special_tokens,
        sampling_strategy=sampling_strategy,
        n_nonzero_tokens_list=n_nonzero_tokens)

    train_loader, train_sampler = init_dataloader_and_sampler(
        cell_dataset=train_cell_dataset,
        batch_size=batch_size,
        distributed=True,
        world_size=world_size,
        rank=rank,
        collate_fn=mask_collator,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=False)

    ipe = len(train_loader)

    # Initialize optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)

    for p in target_encoder.parameters():
        p.requires_grad = False

    # Define momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # Load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch, iter_number = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)

    def save_checkpoint(epoch, iter_number=None):
        save_dict = {'encoder': encoder.state_dict(),
                     'predictor': predictor.state_dict(),
                     'target_encoder': target_encoder.state_dict(),
                     'opt': optimizer.state_dict(),
                     'scaler': None if scaler is None else scaler.state_dict(),
                     'epoch': epoch,
                     'zero_epoch_tracking': True,
                     'loss': loss_meter.avg,
                     'batch_size': batch_size,
                     'world_size': world_size,
                     'lr': lr}
        if iter_number is not None:
            save_dict['iter_number'] = iter_number
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                if iter_number is None:
                    torch.save(save_dict, save_path.format(epoch=f'{epoch}'))
                else:
                    torch.save(save_dict, save_path.format(epoch=f'{epoch}_{iter_number}'))

    # Run training loop
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}")

        # Update distributed dataloader epoch
        train_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred, masks_attention) in enumerate(
        train_loader):
            for key, val in udata.items():
                udata[key] = val.to(device, non_blocking=True)
            masks_enc = [u.to(device, non_blocking=True) for u in masks_enc]
            masks_pred = [u.to(device, non_blocking=True) for u in masks_pred]
            masks_attention = masks_attention.to(device, non_blocking=True)

            if WORLD_RANK == 0 and (itr % log_freq == 0):
                compute_grad_stats = True
            else:
                compute_grad_stats = False

            _new_lr = scheduler.step()
            _new_wd = wd_scheduler.step()

            # Step 1: forward pass
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                # Forward pass of target encoder
                with torch.no_grad():
                    h = target_encoder(
                        batch=udata, masks_attention=masks_attention)
                    h = F.layer_norm(h, (h.size(-1),))
                    h = apply_masks(h, masks_pred, concat=False)

                # Forward pass of context encoder
                z = encoder(batch=udata,
                            masks=masks_enc,
                            masks_attention=None)

                # Forward pass of predictor
                if gt_type == 'rank':
                    z = predictor(z=z,
                                    batch=udata,
                                    masks_enc=masks_enc,
                                    masks_pred=masks_pred,
                                    enc_seg_embed=encoder.module.backbone.seg_embed,
                                    enc_pos_embed=encoder.module.backbone.pos_embed,
                                    masks_attention=None)
                elif gt_type == 'counts':
                    z = predictor(z=z,
                                    batch=udata,
                                    masks_enc=masks_enc,
                                    masks_pred=masks_pred,
                                    enc_seg_embed=encoder.module.backbone.seg_embed,
                                    enc_token_embed=encoder.module.backbone.token_embed,
                                    masks_attention=None)

                # Compute loss
                loss_exp = 1.0
                loss = 0.
                for zi, hi in zip(z, h):
                    if loss_fn_type == 'smooth_l1':
                        loss += F.smooth_l1_loss(zi, hi)
                    elif loss_fn_type == 'l1':
                        loss += torch.mean(
                            torch.abs(zi - hi)**loss_exp) / loss_exp
                loss /= len(masks_pred)

            # Step 2: backward pass and step
            _enc_norm, _pred_norm = 0., 0.
            loss.backward()
            if warmup >= 1: # iteration-based clipping didn't always work # TODO
                if (epoch >= warmup) and (clip_grad is not None):
                    _enc_norm = torch.nn.utils.clip_grad_norm_(
                        encoder.parameters(), clip_grad)
                    _pred_norm = torch.nn.utils.clip_grad_norm_(
                        predictor.parameters(), clip_grad)
            elif (itr > (warmup * ipe)) and (clip_grad is not None):
                _enc_norm = torch.nn.utils.clip_grad_norm_(
                    encoder.parameters(), clip_grad)
                _pred_norm = torch.nn.utils.clip_grad_norm_(
                    predictor.parameters(), clip_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Step 3: momentum update of target encoder
            with torch.no_grad():
                m = next(momentum_scheduler)
                q_params = [p.detach() for p in encoder.parameters()]
                k_params = [p for p in target_encoder.parameters()]
                torch._foreach_mul_(k_params, m)
                tmp = torch._foreach_mul(q_params, (1.0 - m))
                torch._foreach_add_(k_params, tmp)

            # Logging
            if compute_grad_stats:
                grad_stats = grad_logger(encoder.named_parameters())
                grad_stats.global_norm = float(_enc_norm)
                grad_stats_pred = grad_logger(predictor.named_parameters())
                grad_stats_pred.global_norm = float(_pred_norm)
            else:
                grad_stats = None
                grad_stats_pred = None

            loss_meter.update(float(loss))

            #log_stats()
            if WORLD_RANK == 0 and (itr % log_freq == 0):
                wandb.log({
                    "loss": float(loss),
                    "lr": float(_new_lr),
                    "epoch": int(epoch),
                    "global_norm_enc": float(grad_stats.global_norm),
                    "global_norm_pred": float(grad_stats_pred.global_norm),
                })
            assert not np.isnan(float(loss)), 'loss is nan'
            #if itr % checkpoint_freq_iter == 0:
            #    logger.info(f'Saving checkpoint at epoch {epoch} iteration {itr}')
            #    save_checkpoint(epoch, itr // checkpoint_freq_iter)

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch)