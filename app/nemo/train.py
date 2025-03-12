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
                                init_distributed,
                                AllReduceSum)
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
          LOCAL_RANK: Optional[int]=None,
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
    LOCAL_RANK:
        Rank of the process.
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
    elif LOCAL_RANK is not None:
        device = torch.device(f"cuda:{LOCAL_RANK}")
    elif LOCAL_RANK is  None:
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
    n_value_bins = args['meta']['n_value_bins']
    enc_depth = args['meta']['enc_depth'] 
    enc_emb_dim = args['meta']['enc_emb_dim']    
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    special_tokens = args['meta']['special_tokens']
    use_bfloat16 = args['meta']['use_bfloat16']
    use_flash_attention = args['meta']['use_flash_attention']

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
    clip_grad = args['optimization']['clip_grad']

    log_freq = args['state']['log_freq']
    checkpoint_freq = args['state']['checkpoint_freq']
    checkpoint_freq_iter = args['state']['checkpoint_freq_iter']
    write_tag = args['state']['write_tag']
    load_model = args['state']['load_checkpoint'] or resume_preempt
    r_file = args['state']['read_checkpoint']
    load_folder_path = args['state']['folder_path']

    if args['data']['precomputed_n_nonzero_tokens']:
        with open(args['data']['precomputed_n_nonzero_tokens'], "rb") as f: 
            n_nonzero_tokens= pickle.load(f)
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
    
    # Start multiprocessing
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # Define log/checkpointing paths
    log_file = os.path.join(save_folder_path, f'{write_tag}_r{rank}.csv')
    save_path = os.path.join(save_folder_path,
                             f'{write_tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(save_folder_path, f'{write_tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(
            load_folder_path, r_file) if r_file is not None else latest_path

    # Initialize csv logger
    if rank==0:
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
        n_value_bins=n_value_bins,
        device=device,
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_special_tokens=n_special_tokens,
        n_segments=n_segments,
        n_special_values=n_special_values,
        enc_emb_dim=enc_emb_dim,
        enc_depth=enc_depth,
        pred_emb_dim=pred_emb_dim,
        pred_depth=pred_depth,
        use_flash_attention=use_flash_attention)
    target_encoder = copy.deepcopy(encoder)

    # Initialize mask collator
    if block_masking:
       mask_collator = BlockMaskCollator(
            n_targets=n_targets,
            n_contexts=n_contexts,
            n_segments=n_segments,
            seq_len_cell=seq_len_cell,
            seq_len_neighborhood=seq_len_neighborhood,
            n_special_tokens=n_special_tokens,
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
    
    # Initialize train and test datasets, dataloaders and samplers
    train_cell_dataset = make_cell_dataset(
        dataset=train_dataset,
        vocab_size=vocab_size,
        seq_len_cell=seq_len_cell,
        seq_len_neighborhood=seq_len_neighborhood,
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
            mask_collator.step()

    def save_checkpoint(epoch, iter_number=None):
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
                if iter_number is None:
                    torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))
                else:
                    torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}_{iter_number}'))

    # Run training loop
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}")

        # Update distributed dataloader epoch
        train_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        input_var_meter = AverageMeter()
        input_var_min_meter = AverageMeter()
        jepa_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()
        mask_meters = [AverageMeter() for _ in range(len(cfgs_mask))]
        gpu_time_meter = AverageMeter()
        wall_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            try:
                udata, masks_enc, masks_pred = next(loader)
            except Exception:
                logger.info('Exhausted data loaders. Refreshing...')
                loader = iter(unsupervised_loader)
                udata, masks_enc, masks_pred, masks_attention = next(loader)
            assert len(masks_enc) == len(masks_pred), \
                'Currently require num encoder masks = num predictor masks'

            def load_cell_graph():
                # Put each cell graph on the GPU and concatenate along batch dimension
                tokens = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)
                segments = torch.cat([u.to(device, non_blocking=True) for u in udata[1]], dim=0)
                if gt_type == 'rank':
                    positions = torch.cat([u.to(device, non_blocking=True) for u in udata[2]], dim=0)
                elif gt_type == 'counts':
                    values = torch.cat([u.to(device, non_blocking=True) for u in udata[2]], dim=0)

                # Put each mask_enc/mask_pred pair on the GPU and reuse the
                # same mask pair for each cell graph sequence
                _masks_enc, _masks_pred = [], []
                for _me, _mp in zip(masks_enc, masks_pred):
                    _me = _me.to(device, non_blocking=True)
                    _mp = _mp.to(device, non_blocking=True)
                    _me = repeat_interleave_batch(_me, batch_size, repeat=num_clips)
                    _mp = repeat_interleave_batch(_mp, batch_size, repeat=num_clips)
                    _masks_enc.append(_me)
                    _masks_pred.append(_mp)

                if gt_type == 'rank':
                    return (tokens, segments, positions, _masks_enc, _masks_pred)
                elif gt_type == 'counts':
                    return (tokens, segments, values, _masks_enc, _masks_pred)

            if gt_type == 'rank':
                tokens, segments, positions, _masks_enc, _masks_pred = load_cell_graph()
            elif gt_type == 'counts':
                tokens, segments, values, _masks_enc, _masks_pred = load_cell_graph()

            for _i, m in enumerate(mask_meters):
                m.update(masks_enc[_i][0].size(-1))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target(tokens: torch.Tensor,
                                   segments: torch.Tensor,
                                   positions: Optional[torch.Tensor]=None,
                                   values: Optional[torch.Tensor]=None):
                    """
                    Returns list of tensors of shape [BATCH_SIZE, SEQ_LEN, EMBED_DIM], one for each
                    mask_pred.
                    """
                    with torch.no_grad(): # no backward pass for target encoder
                        if gt_type == 'rank':
                            h, _, _, _ = target_encoder(
                                positions=positions
                                segments=segments,
                                tokens=tokens)
                        elif gt_type == 'counts':
                            h, _, _, _ = target_encoder(
                                tokens=tokens,
                                segments=segments,
                                values=values)

                        # Normalize over feature dim
                        h = F.layer_norm(h, (h.size(-1),)) # [BATCH_SIZE, SEQ_LEN, EMBED_DIM]

                        # Only keep encoded tgts (masked genes of h)
                        h = apply_masks(h, masks_pred, concat=False)

                        return h

                def forward_context(h: torch.Tensor,
                                    tokens: torch.Tensor,
                                    segments: torch.Tensor,
                                    positions: Optional[torch.Tensor]=None,
                                    values: Optional[torch.Tensor]=None):
                    # Context encoder forward pass with output dim (BATCH_SIZE,
                    # MIN_CONTEXT_SIZE, EMB_DIM) where MIN_CONTEXT_SIZE is
                    # minmum context size in the batch after removal of
                    # overlapping targets
                    if gt_type == 'rank':
                        z, pos_emb, seg_emb, token_emb = encoder(
                            positions=positions,
                            segments=segments,
                            tokens=tokens,
                            masks=masks_enc)                       
                    elif gt_type == 'counts':
                        z, token_emb, seg_emb, value_emb = encoder(
                            tokens=tokens,
                            segments=segments,
                            counts=counts,
                            masks=masks_enc)

                    # Predictor forward pass with output dim (BATCH_SIZE *
                    # N_TARGETS * N_CONTEXTS, TARGET_MASK_SIZE, EMB_DIM)
                    if gt_type == 'rank':
                        z = predictor(z=z,
                                      pos_embed=pos_emb,
                                      seg_embed=seg_emb,
                                      token_embed=token_emb,
                                      masks_enc=masks_enc,
                                      masks_pred=masks_pred)
                    elif gt_type == 'counts':
                        z = predictor(z=z,
                                      token_embed=token_emb,
                                      seg_embed=seg_emb,
                                      value_embed=value_emb,
                                      masks_enc=masks_enc,
                                      masks_pred=masks_pred)
                    return z

                def loss_fn(z, h):
                    loss = 0.
                    # Compute loss and accumulate for each mask_enc/mask_pred pair
                    for zi, hi in zip(z, h):
                        loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
                    loss /= len(masks_pred)
                    return loss

                def reg_fn(z):
                    return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)

                # Step 1: forward pass
                loss_jepa, loss_reg = 0., 0.
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    if gt_type == 'counts':
                        h = forward_target(
                            tokens=tokens,
                            segments=segments,
                            values=values)
                        z = forward_context(
                            h,
                            tokens=tokens,
                            segments=segments,
                            values=values)
                    loss_jepa = loss_fn(z, h) # jepa prediction loss
                    pstd_z = reg_fn(z) # predictor variance across genes
                    loss_reg += torch.mean(F.relu(1.-pstd_z))
                loss = loss_jepa + reg_coeff * loss_reg

                # Step 2: backward pass and step
                _enc_norm, _pred_norm = 0., 0.
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if (epoch > warmup) and (clip_grad is not None):
                    _enc_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                    _pred_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                grad_stats.global_norm = float(_enc_norm)
                grad_stats_pred = grad_logger(predictor.named_parameters())
                grad_stats_pred.global_norm = float(_pred_norm)
                optimizer.zero_grad()
                optim_stats = adamw_logger(optimizer)

                # Step 3: momentum update of target encoder
                m = next(momentum_scheduler)
                with torch.no_grad():
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (
                    float(loss),
                    float(loss_jepa),
                    float(loss_reg),
                    _new_lr,
                    _new_wd,
                    grad_stats,
                    grad_stats_pred,
                    optim_stats,
                )
            (loss, loss_jepa, loss_reg, _new_lr, _new_wd, grad_stats, grad_stats_pred, optim_stats,), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.
            loss_meter.update(loss)
            input_var = float(AllReduce.apply(clips.view(clips.shape[0], -1).var(dim=1).mean(dim=0)))
            input_var_min = float(AllReduce.apply(torch.min(clips.view(clips.shape[0], -1).var(dim=1))))
            input_var_meter.update(input_var)
            input_var_min_meter.update(input_var_min)
            jepa_loss_meter.update(loss_jepa)
            reg_loss_meter.update(loss_reg)
            gpu_time_meter.update(gpu_etime_ms)
            wall_time_meter.update(iter_elapsed_time_ms)

            # Logging
            def log_stats():
                csv_logger.log(
                    epoch + 1,
                    itr,
                    loss,
                    loss_jepa,
                    loss_reg,
                    grad_stats.global_norm,
                    grad_stats_pred.global_norm,
                    gpu_etime_ms,
                    iter_elapsed_time_ms)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        '[%d, %5d] loss: %.3f | p%.3f r%.3f | '
                        'input_var: %.3f %.3f | '
                        'masks: %s '
                        '[wd: %.2e] [lr: %.2e] '
                        '[mem: %.2e] '
                        '[gpu: %.1f ms]'
                        '[wall: %.1f ms]'
                        % (epoch + 1, itr,
                           loss_meter.avg,
                           jepa_loss_meter.avg,
                           reg_loss_meter.avg,
                           input_var_meter.avg,
                           input_var_min_meter.avg,
                           '[' + ', '.join(['%.1f' % m.avg for m in mask_meters]) + ']',
                           _new_wd,
                           _new_lr,
                           torch.cuda.max_memory_allocated() / 1024.0**2,
                           gpu_time_meter.avg,
                           wall_time_meter.avg))

                    if optim_stats is not None:
                        logger.info(
                            '[%d, %5d] first moment: %.2e [%.2e %.2e] second moment: %.2e [%.2e %.2e]'
                            % (epoch + 1, itr,
                               optim_stats.get('exp_avg').avg,
                               optim_stats.get('exp_avg').min,
                               optim_stats.get('exp_avg').max,
                               optim_stats.get('exp_avg_sq').avg,
                               optim_stats.get('exp_avg_sq').min,
                               optim_stats.get('exp_avg_sq').max))

                    if grad_stats is not None:
                        logger.info(
                            '[%d, %5d] enc_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e'
                            % (epoch + 1, itr,
                               grad_stats.first_layer,
                               grad_stats.last_layer,
                               grad_stats.min,
                               grad_stats.max,
                               grad_stats.global_norm))

                    if grad_stats_pred is not None:
                        logger.info(
                            '[%d, %5d] pred_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e'
                            % (epoch + 1, itr,
                               grad_stats_pred.first_layer,
                               grad_stats_pred.last_layer,
                               grad_stats_pred.min,
                               grad_stats_pred.max,
                               grad_stats_pred.global_norm))
            log_stats()
            assert not np.isnan(loss), 'loss is nan'
            wandb.log(
                {"loss": loss,
                'lr':_new_lr,
                'epoch': epoch,
                'global_norm_enc': grad_stats.global_norm,
                'global_norm_pred': grad_stats_pred.global_norm,
                })

            # Save checkpoint after specified number of iterations
            if itr % checkpoint_freq_iter == 0:
                logger.info(f'Saving checkpoint at epoch {epoch} iteration {itr}')
                save_checkpoint(epoch + 1, itr // checkpoint_freq_iter)

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)
