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
import multiprocessing as mp
import logging
import pickle
import random
import sys
import yaml

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from .datasets.cell_neighborhood_dataset import (CellNeighborhoodDataset,
                                                 make_cell_neighborhood_dataset)
from .helper import (init_model,
                     init_opt,
                     load_checkpoint)
from .masks.multigene import MaskCollator
from .masks.segment_masking  import SegmentMaskCollator
from .masks.utils import apply_masks
from .utils.distributed import (AllReduce,
                                init_distributed)
from .utils.logging import (AverageMeter,
                            CSVLogger,
                            gpu_timer,
                            grad_logger)
from .utils.tensors import repeat_interleave_batch, clip_gradients


log_timings = True
log_freq = 10
checkpoint_freq = 5


_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def train(args: dict,
          train_dataset: CellNeighborhoodDataset,
          test_dataset: CellNeighborhoodDataset,
          resume_preempt: bool=False
          ):
    """
    Train the model.

    Parameters
    -----------
    args:
        Dictionary containing the hyperparams from the config file.
    train_dataset:
        Train split CellNeighborhoodDataset.
    test_dataset:
        Test split CellNeighborhoodDataset.
    resume_preempt:
    """
    # ----------------------------- #
    #  Load params from config file
    # ----------------------------- #

    # Load meta params
    use_bfloat16 = args['meta']['use_bfloat16']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    enc_depth = args['meta']['enc_depth'] 
    enc_emb_dim = args['meta']['enc_emb_dim']    
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    pos_learnable = args['meta']['pos_learnable']
    seg_learnable = args['meta']['seg_learnable']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # Load data params
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    seq_len_cell = args['data']['seq_len_cell']
    seq_len_neighborhood = args['data']['seq_len_neighborhood']
    has_cls = args['data']['has_cls']
    data_set_name = args['data']['data_set_name']
    vocab_size = args['data']['vocab_size']

    # Load mask params
    n_targets = args['mask']['n_targets']
    n_contexts = args['mask']['n_contexts']
    target_mask_size = args['mask']['target_mask_size']
    context_mask_size = args['mask']['context_mask_size']
    segment_masking = args['mask']['segment_masking']
    per_segment_mask_ratio = args['mask']['per_segment_mask_ratio']

    # Load optimization params
    if isinstance(args['optimization']['ema'], list):
       ema = args['optimization']['ema']
    else:
       ema = [args['optimization']['ema'], 1]
    ipe_scale = args['optimization']['ipe_scale'] # scheduler scale factor
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']
    clip_grad = args['optimization']['clip_grad']
    grad_accum_steps = args['optimization']['grad_accum_steps']
    use_flash_attention = args['optimization']['use_flash_attention']
    
    seq_len = seq_len_cell + seq_len_neighborhood

    # Create folder to store artifacts
    # -- LOGGING
    folder = (f"logs/{data_set_name}_"
              f"pred_depth_{pred_depth}_pred_emb_dim_{pred_emb_dim}_"
              f"enc_depth_{enc_depth}_enc_emb_dim_{enc_emb_dim}_n_targets_{n_targets}_"
              f"n_contexts_{n_contexts}_target_mask_size_{target_mask_size}_"
              f"context_mask_size_{context_mask_size}_num_epochs_{num_epochs}_"
              f"seq_len_cell_{seq_len_cell}_"
              f"seq_len_neighborhood_{seq_len_neighborhood}_"
              f"pos_learnable_{pos_learnable}_"
              f"seg_learnable_{seg_learnable}_"
              f"ratio_{per_segment_mask_ratio}")

    os.makedirs(folder, exist_ok=True)
    tag = args['logging']['write_tag']

    dump = os.path.join(folder, 'params.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)

    # ----------------------------- #
    
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
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(
            folder, r_file) if r_file is not None else latest_path

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
        device=device,
        vocab_size=vocab_size,
        seq_len=seq_len,
        enc_emb_dim=enc_emb_dim,
        enc_depth=enc_depth,
        pred_emb_dim=pred_emb_dim,
        pred_depth=pred_depth,
        pos_learnable=pos_learnable,
        seg_learnable=seg_learnable,
        has_cls=has_cls,
        use_flash_attention=use_flash_attention)
    target_encoder = copy.deepcopy(encoder)

    # Initialize mask collator
    if segment_masking:
       mask_collator = SegmentMaskCollator(
            n_targets=n_targets,
            n_contexts=n_contexts,
            seq_len_cell=seq_len_cell,
            seq_len_neighborhood=seq_len_neighborhood,
            has_cls=has_cls,
            per_segment_mask_ratio = per_segment_mask_ratio)
    else:
        mask_collator = MaskCollator(
            n_targets=n_targets,
            n_contexts=n_contexts,
            target_mask_size=target_mask_size,
            context_mask_size=context_mask_size,
            seq_len_cell=seq_len_cell,
            seq_len_neighborhood=seq_len_neighborhood,
            has_cls=has_cls)
    
    # Initialize dataloader and -sampler
    _, train_loader, train_sampler = make_cell_neighborhood_dataset(
        batch_size=batch_size,
        data=train_dataset,
        vocab_size=vocab_size,
        collator=mask_collator,
        pin_mem=pin_mem,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        drop_last=False,
        seq_len_cell=seq_len_cell,
        seq_len_neighborhood=seq_len_neighborhood,
        has_cls=has_cls)

    _, test_loader, test_sampler = make_cell_neighborhood_dataset(
        batch_size=batch_size,
        data=test_dataset,
        vocab_size=vocab_size,
        collator=mask_collator,
        pin_mem=pin_mem,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        drop_last=False,
        seq_len_cell=seq_len_cell,
        seq_len_neighborhood=seq_len_neighborhood,
        has_cls=has_cls)

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
    # -- momentum schedule
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
            mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
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

        for itr, (udata, masks_enc, masks_pred,  masks_attention) in enumerate(train_loader):
            def load_cell_neighborhoods():
                # -- unsupervised imgs
                cell_neighborhood_tokens = udata[0].to(device,
                                                       non_blocking=True)
                seg_label = udata[1].to(device, non_blocking=True)                 
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                masks_3 = masks_attention.to(device, non_blocking=True)
                return (cell_neighborhood_tokens, seg_label,  masks_1, masks_2, masks_3)
            cell_neighborhood_tokens, seg_label, masks_enc, masks_pred, masks_attention = load_cell_neighborhoods()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target():
                    with torch.no_grad(): # no backward pass for target encoder
                        # Encode all cell neighborhood tokens
                        h = target_encoder(
                            cell_neighborhood_tokens,
                            seg_label,
                            masks_attention=masks_attention) # output (BATCH_SIZE, SEQ_LEN, EMBED_DIM)
                                       # if no <cls> token (BATCH_SIZE,
                                       # SEQ_LEN+1, EMBED_DIM) otherwise
                                       # masks_attention
                        # Normalize over feature dim
                        h = F.layer_norm(h, (h.size(-1),))
                        # Only keep encoded targets (masked genes of h)
                        h = apply_masks(
                            h,
                            masks_pred) # output (BATCH_SIZE * N_TARGETS,
                                        # TARGET_MASK_SIZE, EMB_SIZE)
                        B = len(h)
                        # Repeat targets if multiple contexts
                        h = repeat_interleave_batch(
                            h,
                            B,
                            repeat=len(masks_enc)) # output (BATCH_SIZE *
                                                   # N_TARGETS * N_CONTEXTS,
                                                   # TARGET_MASK_SIZE, EMB_DIM)
                        return h

                def forward_context():
                    # Encode only context cell neighborhood tokens
                    z = encoder(
                        cell_neighborhood_tokens,
                        seg_label,
                        masks_enc) # output (BATCH_SIZE, MIN_CONTEXT_SIZE,
                                   # EMB_DIM) where MIN_CONTEXT_SIZE is minmum
                                   # context size in the batch after removal of
                                   # overlapping targets
                    z = predictor(
                        z,
                        seg_label,
                        masks_enc,
                        masks_pred) # output (BATCH_SIZE * N_TARGETS *
                                    # N_CONTEXTS, TARGET_MASK_SIZE, EMB_DIM)
                    return z

                def loss_fn(z, h):
                    #loss = F.l1_loss(z, h)
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                # Step 1: forward pass
                with torch.cuda.amp.autocast(dtype=torch.bfloat16,
                                             enabled=use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss = loss_fn(z, h)
                    if grad_accum_steps:
                        loss=loss/grad_accum_steps

                # Step 2: backward pass and step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    if clip_grad:
                       scaler.unscale_(optimizer)
                       clip_gradients(encoder, clip_grad)
                       clip_gradients(predictor, clip_grad)
                    if itr % grad_accum_steps == 0:
                       scaler.step(optimizer)
                       scaler.update()
                else:
                    loss.backward()
                    if clip_grad:
                      clip_gradients(encoder, clip_grad)
                      clip_gradients(predictor, clip_grad)
                    if itr % grad_accum_steps == 0:
                       optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                if itr % grad_accum_steps == 0:
                   optimizer.zero_grad()

                # Step 3: momentum update of target encoder
                if itr % grad_accum_steps == 0:
                    with torch.no_grad():
                      m = next(momentum_scheduler)
                      for param_q, param_k in zip(encoder.parameters(),
                                                target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
                
                return (float(loss*grad_accum_steps), _new_lr, _new_wd, grad_stats)
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
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
            log_stats()
            wandb.log({"loss": loss, 'lr':_new_lr})
            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)
if __name__ == "__main__":
    main()
