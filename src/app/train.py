"""
Adapted from Assran, M. et al. Self-supervised learning from images with a 
Joint-Embedding Predictive Architecture. Proc. IEEE Comput. Soc. Conf. Comput.
Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/train.py (05.06.2024).
"""

import os

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
import pickle
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.profiler
import wandb
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from app.helper import init_model, init_opt, load_checkpoint
from nichejepa.datasets_new.cell_datasets import init_cell_dataset
from nichejepa.datasets_new.dataloaders import init_dataloader_and_sampler
from nichejepa.masks_new.block_masking  import BlockMaskCollator
from nichejepa.masks_new.cell_masking import CellMaskCollator
from nichejepa.masks_new.utils import apply_masks
from nichejepa.models_new.utils import repeat_interleave_batch
from nichejepa.utils_new.distributed import init_distributed
from nichejepa.utils_new.logging import (AverageMeter,
                                     CSVLogger,
                                     grad_logger)

os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1" # Better error propagation

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
    token_dict_folder_path = args['data']['token_dict_folder_path']
    tokenizer_type = args['data']['tokenizer_type']
    seq_len_cell = args['data']['seq_len_cell']
    seq_len_neighborhood = args['data']['seq_len_neighborhood']
    n_segments = args['data']['n_segments']
    sampling_strategy = args['data']['sampling_strategy']
    batch_size = args['data']['batch_size']
    num_workers = args['data']['num_workers']
    pin_memory = args['data']['pin_memory']

    if 'sep_gene_tokens_neb' in args['data'].keys():
        sep_gene_tokens_neb = args['data']['sep_gene_tokens_neb']
    else:
        sep_gene_tokens_neb = False

    if 'nz_spc' in args['data'].keys():
        nz_spc = args['data']['nz_spc']
    else:
        nz_spc = False

    if 'use_sampler' in args['data'].keys():
        use_sampler = args['data']['use_sampler']
    else:
        use_sampler = False

    add_cls = args['meta']['add_cls']
    gt_type = args['meta']['gt_type']
    count_encoding = args['meta']['count_encoding']
    n_value_bins = args['meta']['n_value_bins']
    if 'cell_pos_enc' in args['meta'].keys():
        cell_pos_enc = args['meta']['cell_pos_enc']
    else:
        cell_pos_enc = 'segment'
    enc_depth = args['meta']['enc_depth'] 
    enc_emb_dim = args['meta']['enc_emb_dim']    
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if 'num_heads' in args['meta'].keys():
        num_heads = args['meta']['num_heads']
    else:
        num_heads = 8
    if 'mlp_ratio' in args['meta'].keys():
        mlp_ratio = args['meta']['mlp_ratio']
    else:
        mlp_ratio = 4.0
    if 'loss_fn_type' in args['meta'].keys():
        loss_fn_type = args['meta']['loss_fn_type']
    else:
        loss_fn_type = 'l1'
    if 'predict_gene' in args['meta'].keys():
        predict_gene = args['meta']['predict_gene']
    else:
        predict_gene = True
    if 'pos_learnable' in args['meta'].keys():
        pos_learnable = args['meta']['pos_learnable']
    else:
        pos_learnable = False
    special_tokens = args['meta']['special_tokens']
    use_bfloat16 = args['meta']['use_bfloat16']
    use_flash_attention = args['meta']['use_flash_attention']
    use_layer_norm = args['meta']['use_layer_norm']

    n_contexts = args['mask']['n_contexts']
    n_targets = args['mask']['n_targets']
    block_masking = args['mask']['block_masking']
    cell_masking = args['mask']['cell_masking']
    context_mask_size = args['mask']['context_mask_size']
    target_mask_size = args['mask']['target_mask_size']
    per_block_mask_ratio = args['mask']['per_block_mask_ratio']
    if 'sample_segments' in args['mask'].keys():
        sample_segments = args['mask']['sample_segments']
    else:
        sample_segments = False
    targets_list = args['mask']['targets_list']

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
    if 'lambda_var' in args['optimization'].keys():
        lambda_var = args['optimization']['lambda_var']
    else:
        lambda_var = 0.
    if 'lambda_cov' in args['optimization'].keys():
        lambda_cov = args['optimization']['lambda_cov']
    else:
        lambda_cov = 0.

    log_freq = args['state']['log_freq']
    checkpoint_freq = args['state']['checkpoint_freq']
    checkpoint_freq_iter = args['state']['checkpoint_freq_iter']
    write_tag = args['state']['write_tag']
    load_model = args['state']['load_checkpoint'] or resume_preempt
    r_file = args['state']['read_checkpoint']
    load_folder_path = args['state']['folder_path']
    use_profiler = args['state'].get('use_profiler', False)

    if 'precomputed_epoch_n_nonzero_tokens' in args['data'].keys():
        with open(args['data']['precomputed_epoch_n_nonzero_tokens'], "rb") as f: 
            epoch_n_nonzero_tokens = pickle.load(f)
    elif args['data']['precomputed_n_nonzero_tokens']:
        with open(args['data']['precomputed_n_nonzero_tokens'], "rb") as f: 
            n_nonzero_tokens = pickle.load(f)
    else:
        n_nonzero_tokens = None

    # Load token dict and get token dict-specfic params
    with open(token_dict_folder_path, 'rb') as file:
        token_dict = pickle.load(file)
    vocab_size = len(token_dict)
    #n_special_values = sum(
    #    1 for key in token_dict if "spv" in key) # this only works now because of the dummy special values
    n_special_values = args['data']['n_special_values']
    max_special_tokens = sum(
        1 for key in token_dict if "cls" in key) + sum(
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

    # Set multiprocessing start method
    try:
        mp.set_start_method("spawn")
    except Exception:
        logger.info(f'Multiprocessing not started.')
    
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

    # Initialize csv logger
    #if rank==0:
    #    csv_logger = CSVLogger(log_file,
    #                       ('%d', 'epoch'),
    #                       ('%d', 'itr'),
    #                       ('%.5f', 'loss'),
    #                       ('%.5f', 'mask-A'),
    #                       ('%.5f', 'mask-B'),
    #                       ('%d', 'time (ms)'))

    # Initialize encoder, predictor and target encoder
    encoder, predictor = init_model(
        gt_type=gt_type,
        count_encoding=count_encoding,
        n_value_bins=n_value_bins,
        cell_pos_enc=cell_pos_enc,
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
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        use_flash_attention=use_flash_attention,
        use_layer_norm=use_layer_norm,
        sep_gene_tokens_neb=sep_gene_tokens_neb,
        predict_gene=predict_gene,
        pos_learnable=pos_learnable,
        nz_spc=nz_spc)
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
            per_block_mask_ratio=per_block_mask_ratio,
            sample_segments=sample_segments,
            sample_gene_masks=True)
    elif cell_masking:
       mask_collator = CellMaskCollator(
            n_targets=n_targets,
            n_contexts=n_contexts,
            n_segments=n_segments,
            seq_len_cell=seq_len_cell,
            seq_len_neighborhood=seq_len_neighborhood,
            n_special_tokens=n_special_tokens,
            per_block_mask_ratio=per_block_mask_ratio,
            targets_list=targets_list)
    
    # Initialize train and test datasets, dataloaders and samplers
    if isinstance(train_dataset, list):
        train_cell_datasets = []
        for d, nz in zip(train_dataset, epoch_n_nonzero_tokens):
            cell_d = init_cell_dataset(
                dataset=d,
                vocab_size=vocab_size,
                seq_len_cell=seq_len_cell,
                seq_len_neighborhood=seq_len_neighborhood,
                tokenizer_type=tokenizer_type,
                gt_type=gt_type,
                cell_pos_enc=cell_pos_enc,
                special_tokens=special_tokens,
                sampling_strategy=sampling_strategy,
                n_nonzero_tokens_list=nz,
                include_cell_id=False,
                sep_gene_tokens_neb=sep_gene_tokens_neb,
                nz_spc=nz_spc)
            train_cell_datasets.append(cell_d)

    else:
        train_cell_dataset = init_cell_dataset(
            dataset=train_dataset,
            vocab_size=vocab_size,
            seq_len_cell=seq_len_cell,
            seq_len_neighborhood=seq_len_neighborhood,
            tokenizer_type=tokenizer_type,
            gt_type=gt_type,
            cell_pos_enc=cell_pos_enc,
            special_tokens=special_tokens,
            sampling_strategy=sampling_strategy,
            n_nonzero_tokens_list=n_nonzero_tokens,
            include_cell_id=False,
            sep_gene_tokens_neb=sep_gene_tokens_neb,
            nz_spc=nz_spc)

    if isinstance(train_dataset, list):
        train_loaders = []
        train_samplers = []
        for cell_d in train_cell_datasets:
            train_loader, train_sampler = init_dataloader_and_sampler(
                cell_dataset=cell_d,
                batch_size=batch_size,
                distributed=use_sampler,
                world_size=world_size,
                rank=rank,
                collate_fn=mask_collator,
                pin_memory=pin_memory,
                num_workers=num_workers,
                drop_last=True,
                prefetch_factor=(4 if num_workers > 0 else None),
                persistent_workers=False)
            train_loaders.append(train_loader)
            train_samplers.append(train_sampler)

    else:
        train_loader, train_sampler = init_dataloader_and_sampler(
            cell_dataset=train_cell_dataset,
            batch_size=batch_size,
            distributed=use_sampler,
            world_size=world_size,
            rank=rank,
            collate_fn=mask_collator,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=True,
            prefetch_factor=(4 if num_workers > 0 else None),
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
    
    #encoder = torch.compile(encoder)
    #predictor = torch.compile(predictor)

    if use_profiler and WORLD_RANK == 0:

        def trace_handler(p):
            cpu_output = p.key_averages(group_by_stack_n=5).table(
                sort_by="cpu_time_total", row_limit=100)
            logger.info(f"Profiler CPU output: {cpu_output}.")
            gpu_output = p.key_averages(group_by_stack_n=5).table(
                sort_by="cuda_time_total", row_limit=100)
            logger.info(f"Profiler CPU output: {cpu_output}.")
            logger.info(f"Profiler GPU output: {gpu_output}.")
            p.export_chrome_trace(
                os.path.join(
                    save_folder_path,
                    "profiler_logs/trace_" + str(p.step_num) + ".json"))

        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=10,
                warmup=1,
                active=3,
                repeat=1),
            on_trace_ready=trace_handler,
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
        )
        profiler.start()

    encoder = DistributedDataParallel(
        encoder,
        static_graph=True,
        device_ids=[LOCAL_RANK],
        output_device=LOCAL_RANK,
        gradient_as_bucket_view=True,
        broadcast_buffers=False)
    predictor = DistributedDataParallel(
        predictor,
        static_graph=True,
        device_ids=[LOCAL_RANK],
        output_device=LOCAL_RANK,
        gradient_as_bucket_view=True,
        broadcast_buffers=False)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # Define momentum schedule
    momentum_scheduler = (
        ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
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
            #mask_collator.step()

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
        logger.info(f"Epoch {epoch}")
        if isinstance(train_dataset, list):
            logger.info(f"Using train loader and sampler from epoch {epoch}.")
            train_loader = train_loaders[epoch]
            train_sampler = train_samplers[epoch] 

        # Update distributed dataloader epoch
        train_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        #maskA_meter = AverageMeter()
        #maskB_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred, masks_attention) in enumerate(train_loader):
            for key, val in udata.items():
                udata[key] = val.to(device, non_blocking=True)
            masks_enc = [u.to(device, non_blocking=True) for u in masks_enc]
            masks_pred = [u.to(device, non_blocking=True) for u in masks_pred]
            masks_attention = masks_attention.to(device, non_blocking=True)

            assert len(masks_enc) == 1, 'Currently require num encoder masks = 1'

            #maskA_meter.update(len(masks_enc[0][0]))
            #maskB_meter.update(len(masks_pred[0][0]))

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
                    h, _ = target_encoder(
                        batch=udata, masks_attention=masks_attention)
                    h = F.layer_norm(h, (h.size(-1),))
                    h = apply_masks(h, masks_pred, concat=False)

                # Forward pass of context encoder
                z, token_emb = encoder(
                    batch=udata,
                    masks=masks_enc,
                    masks_attention=None)

                # Forward pass of predictor
                z = predictor(
                    z=z,
                    token_emb=token_emb,
                    batch=udata,
                    masks_enc=masks_enc,
                    masks_pred=masks_pred,
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

                # ----------------------------------------------------
                # VICReg-style variance + covariance regularization
                # ----------------------------------------------------

                all_z = torch.cat(z, dim=0)
                all_z = all_z.reshape(-1, all_z.shape[-1])

                # ----- variance loss -----
                def variance_loss(z, gamma=1.0, eps=1e-4):
                    std = z.std(dim=0) + eps
                    return torch.mean(torch.relu(gamma - std))

                # ----- covariance loss -----
                def covariance_loss(z):
                    z = z - z.mean(dim=0)
                    N, D = z.shape
                    cov = (z.T @ z) / (N - 1)
                    off_diag = cov - torch.diag(torch.diag(cov))
                    return (off_diag**2).sum() / D

                loss_var = variance_loss(all_z)
                loss_cov = covariance_loss(all_z)

                loss = loss + lambda_var * loss_var + lambda_cov * loss_cov

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

            if use_profiler and WORLD_RANK == 0:
                profiler.step()

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

            # Logging
            #def log_stats():
            #    csv_logger.log(epoch,
            #                   itr,
            #                   loss,
            #                   grad_stats.global_norm,
            #                   grad_stats_pred.global_norm,
            #                   maskA_meter.val,
            #                   maskB_meter.val,
            #                   etime)
            #    if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
            #        logger.info('[%d, %5d] loss: %.3f '
            #                    'masks: %.1f %.1f '
            #                    '[wd: %.2e] [lr: %.2e] '
            #                    '[mem: %.2e] '
            #                    '(%.1f ms)'
            #                    % (epoch, itr,
            #                       loss_meter.avg,
            #                       maskA_meter.avg,
            #                       maskB_meter.avg,
            #                       _new_wd,
            #                       _new_lr,
            #                       torch.cuda.max_memory_allocated() / 1024.**2,
            #                       time_meter.avg))
            #
            #        if grad_stats is not None:
            #            logger.info(
            #                '[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
            #                % (epoch, itr,
            #                grad_stats.first_layer,
            #                grad_stats.last_layer,
            #                grad_stats.min,
            #                grad_stats.max))

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
            if itr % checkpoint_freq_iter == 0:
                logger.info(f'Saving checkpoint at epoch {epoch} iteration {itr}')
                save_checkpoint(epoch, itr // checkpoint_freq_iter)

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch)

    if use_profiler and WORLD_RANK == 0 :
        # Close profiler
        profiler.stop()