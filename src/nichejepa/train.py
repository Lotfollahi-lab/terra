"""
Adapted from Assran, M. et al. Self-supervised learning from images with a 
Joint-Embedding Predictive Architecture. Proc. IEEE Comput. Soc. Conf. Comput.
Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/train.py (05.06.2024).
"""

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml
from datetime import datetime
from typing import Optional

import datasets
import numpy as np
import pandas as pd
import pickle
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from .datasets.cell_datasets import make_cell_dataset
from .datasets.dataloaders import init_dataloader_and_sampler
from .helper import (init_model,
                     init_opt,
                     load_checkpoint)
from .masks.random_masking import RandomMaskCollator
from .masks.block_masking  import BlockMaskCollator
from .masks.utils import apply_masks
from .models.utils import repeat_interleave_batch
from .utils.distributed import (AllReduce,
                                init_distributed)
from .utils.logging import (AverageMeter,
                            CSVLogger,
                            gpu_timer,
                            grad_logger)

_GLOBAL_SEED = 0


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def train(args: dict,
          train_dataset: datasets.Dataset,
          resume_preempt: bool=False,
          save_folder_path: Optional[str]=None,
          ):
    """
    Train model.

    Parameters
    -----------
    args:
        Dictionary containing the hyperparams from the config file.
    train_dataset:
        Train split of huggingface dataset.
    resume_preempt:
    save_folder_path:
        Path for saving model artifacts.
    """
    # Set random seeds
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # Load params from config file
    dataset_name = args['data']['dataset_name']
    token_dict_folder_path = args['data']['token_dict_folder_path']
    tokenizer_type = args['data']['tokenizer_type']
    seq_len_cell = args['data']['seq_len_cell']
    seq_len_neighborhood = args['data']['seq_len_neighborhood']
    n_segments = args['data']['n_segments']
    sampling_strategy = args['data']['sampling_strategy']
    batch_size = args['data']['batch_size']
    num_workers = args['data']['num_workers']
    pin_memory = args['data']['pin_memory']

    add_cls = args['meta']['add_cls']
    gt_type = args['meta']['gt_type']
    enc_depth = args['meta']['enc_depth'] 
    enc_emb_dim = args['meta']['enc_emb_dim']    
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    special_tokens = args['meta']['special_tokens']
    pos_learnable = args['meta']['pos_learnable']
    seg_learnable = args['meta']['seg_learnable']
    use_bfloat16 = args['meta']['use_bfloat16']
    use_flash_attention = args['meta']['use_flash_attention']
    centering = args['meta']['centering']
    center_momentum = args['meta']['center_momentum']

    n_contexts = args['mask']['n_contexts']
    n_targets = args['mask']['n_targets']
    block_masking = args['mask']['block_masking']
    context_mask_size = args['mask']['context_mask_size']
    target_mask_size = args['mask']['target_mask_size']
    per_block_mask_ratio = args['mask']['per_block_mask_ratio']

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

    log_freq = args['state']['log_freq']
    checkpoint_freq = args['state']['checkpoint_freq']
    write_tag = args['state']['write_tag']
    load_model = args['state']['load_checkpoint'] or resume_preempt
    r_file = args['state']['read_checkpoint']

    if args['data']['precomputed_n_nonzero_tokens']:
        with open(args['data']['precomputed_n_nonzero_tokens'], "rb") as f:
            n_nonzero_tokens = pickle.load(f)
    else:
        n_nonzero_tokens = None
        print(n_nonzero_tokens)

    # Load token dict and get token dict-specfic params
    with open(token_dict_folder_path, 'rb') as file:
        token_dict = pickle.load(file)
    vocab_size = len(token_dict)
    n_special_values = sum(1 for key in token_dict if "spv" in key)
    max_special_tokens = sum(1 for key in token_dict if "cls" in key) + sum(
        1 for key in token_dict if "spt" in key)

    # Define tokenizer-specific params
    if tokenizer_type == 'cell_neighborhood':
        if add_cls:
            special_tokens = ['cls_0', 'cls_1'] + special_tokens            
    elif tokenizer_type == 'cell_graph':
        if add_cls:
            special_tokens = [
                f'cls_{i}' for i in range(n_segments)] + special_tokens
    
    max_cls_tokens = sum('cls' in token for token in special_tokens)

    # Get token sequence length and number of special tokens
    n_special_tokens = len(special_tokens)
    seq_len = seq_len_cell + seq_len_neighborhood + n_special_tokens

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
    os.makedirs(save_folder_path, exist_ok=True)

    # Store config file with model
    dump = os.path.join(save_folder_path, 'params.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    
    # Start multiprocessing
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass
    
    # Initialize torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}.')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # Define log/checkpointing paths
    log_file = os.path.join(save_folder_path, f'{write_tag}_r{rank}.csv')
    save_path = os.path.join(save_folder_path,
                             f'{write_tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(save_folder_path, f'{write_tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(
            save_folder_path, r_file) if r_file is not None else latest_path

    # Initialize csv logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

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
        use_flash_attention=use_flash_attention)
    target_encoder = copy.deepcopy(encoder)

    # Initialize center for target centering
    center = None

    # Initialize mask collator
    if block_masking:
       mask_collator = BlockMaskCollator(
            n_targets=n_targets,
            n_contexts=n_contexts,
            n_segments=n_segments,
            seq_len_cell=seq_len_cell,
            seq_len_neighborhood=seq_len_neighborhood,
            max_special_tokens=max_special_tokens,
            n_special_tokens=n_special_tokens,
            max_cls_tokens=max_cls_tokens,
            per_block_mask_ratio=per_block_mask_ratio)
    else:
        mask_collator = RandomMaskCollator(
            n_targets=n_targets,
            n_contexts=n_contexts,
            seq_len_cell=seq_len_cell,
            seq_len_neighborhood=seq_len_neighborhood,
            n_special_tokens=n_special_tokens,
            target_mask_size=target_mask_size,
            context_mask_size=context_mask_size,)
    
    # Initialize train datasets, dataloaders and samplers
    train_cell_dataset = make_cell_dataset(
        dataset=train_dataset,
        vocab_size=vocab_size,
        seq_len_cell=seq_len_cell,
        seq_len_neighborhood=seq_len_neighborhood,
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
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # Define momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # Load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
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

    def save_checkpoint(epoch):
        save_dict = {'encoder': encoder.state_dict(),
                     'predictor': predictor.state_dict(),
                     'target_encoder': target_encoder.state_dict(),
                     'opt': optimizer.state_dict(),
                     'scaler': None if scaler is None else scaler.state_dict(),
                     'epoch': epoch,
                     'loss': loss_meter.avg,
                     'batch_size': batch_size,
                     'world_size': world_size,
                     'lr': lr}
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # Run training loop
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}")

        # Update distributed dataloader epoch
        train_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred, masks_attention) in enumerate(
        train_loader):
            tokens = udata[0].to(device, non_blocking=True)
            segments = udata[1].to(device, non_blocking=True)
            if gt_type == 'rank':
                positions = udata[2].to(device, non_blocking=True)
            elif gt_type == 'counts':
                counts = udata[2].to(device, non_blocking=True)
            masks_enc = [u.to(device, non_blocking=True) for u in masks_enc]
            masks_pred = [u.to(device, non_blocking=True) for u in masks_pred]
            masks_attention = masks_attention.to(device, non_blocking=True)

            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step(center):
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target(center):
                    with torch.no_grad(): # no backward pass for target encoder
                        # Target encorder forward pass with output dim 
                        # (BATCH_SIZE, SEQ_LEN, EMBED_DIM)
                        if gt_type == 'rank':
                            h, _, _, _ = target_encoder(tokens=tokens,
                                               segments=segments,
                                               positions=positions,
                                               masks_attention=masks_attention)
                        elif gt_type == 'counts':
                            h, _, _, _ = target_encoder(tokens=tokens,
                                               segments=segments,
                                               counts=counts,
                                               masks_attention=masks_attention)

                        if centering:
                            # Update center over batch for centering like in DINO
                            if center is not None:
                                center = center_momentum * center + (
                                    1-center_momentum) * h.mean(
                                        dim=0, keepdim=True)
                            else:
                                center = h.mean(dim=0, keepdim=True)
                            
                            # Center over batch
                            h = h - center
                        else:
                            # Normalize over feature dim
                            h = F.layer_norm(h, (h.size(-1),))

                        # Only keep encoded targets (masked genes of h); output
                        # dim (BATCH_SIZE * N_TARGETS, TARGET_MASK_SIZE, 
                        # EMB_SIZE)
                        h = apply_masks(
                            h,
                            masks_pred)
                        B = len(h)

                        # Repeat targets if multiple contexts; output dim 
                        # (BATCH_SIZE * N_TARGETS * N_CONTEXTS, 
                        # TARGET_MASK_SIZE, EMB_DIM)
                        h = repeat_interleave_batch(
                            h,
                            B,
                            repeat=len(masks_enc))

                        return h, center

                def forward_context():
                    # Context encoder forward pass with output dim (BATCH_SIZE,
                    # MIN_CONTEXT_SIZE, EMB_DIM) where MIN_CONTEXT_SIZE is
                    # minmum context size in the batch after removal of
                    # overlapping targets
                    if gt_type == 'rank':
                        z, pos_emb, seg_emb, token_emb = encoder(
                            positions=positions,
                            segments=segments,
                            tokens=tokens,
                            masks=masks_enc,
                            masks_attention=None)                       
                    elif gt_type == 'counts':
                        z, token_emb, seg_emb, value_emb = encoder(
                            tokens=tokens,
                            segments=segments,
                            counts=counts,
                            masks=masks_enc,
                            masks_attention=None)

                    # Predictor forward pass with output dim (BATCH_SIZE *
                    # N_TARGETS * N_CONTEXTS, TARGET_MASK_SIZE, EMB_DIM)
                    if gt_type == 'rank':
                        z = predictor(z=z,
                                      pos_embed=pos_emb,
                                      seg_embed=seg_emb,
                                      token_embed=token_emb,
                                      masks_enc=masks_enc,
                                      masks_pred=masks_pred,
                                      masks_attention=None)
                    elif gt_type == 'counts':
                        z = predictor(z=z,
                                      token_embed=token_emb,
                                      seg_embed=seg_emb,
                                      value_embed=value_emb,
                                      masks_enc=masks_enc,
                                      masks_pred=masks_pred,
                                      masks_attention=None)
                    return z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                # Step 1: forward pass
                with torch.cuda.amp.autocast(dtype=torch.bfloat16,
                                             enabled=use_bfloat16):
                    h, center = forward_target(center)
                    z = forward_context()
                    loss = loss_fn(z, h)

                # Step 2: backward pass and step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3: momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(),
                                                target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats, center)
            (loss, _new_lr, _new_wd, grad_stats, center), etime = gpu_timer(
                train_step, center)
            loss_meter.update(loss)
            time_meter.update(etime)

            # Logging
            def log_stats():
                csv_logger.log(epoch + 1,
                               itr,
                               loss,
                               maskA_meter.val,
                               maskB_meter.val,
                               etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info(
                            '[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                            % (epoch + 1, itr,
                            grad_stats.first_layer,
                            grad_stats.last_layer,
                            grad_stats.min,
                            grad_stats.max))
            #log_stats()
            wandb.log({"loss": loss, 'lr':_new_lr, "epoch": epoch})
            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)
