"""
Adapted from Assran, M. et al. Self-supervised learning from images with a
Joint-Embedding Predictive Architecture. Proc. IEEE Comput. Soc. Conf. Comput.
Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/train.py (05.06.2024).
"""

# Standard library imports
import copy
import logging
import os
import pickle
import sys
from datetime import datetime
from typing import Optional, List
import asyncio
from src.nichejepa.utils.copy_artifacts_from_tmp import copy_artifacts_async

# Third-party imports
import datasets
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
import yaml
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import torch.distributed as dist

# Local imports
from .datasets.cell_datasets import make_cell_dataset
from .datasets.dataloaders import init_dataloader_and_sampler
from .helper import init_model, init_opt, load_checkpoint
from .masks.block_masking import BlockMaskCollator
from .masks.cell_masking import CelllMaskCollator
from .masks.random_masking import RandomMaskCollator
from .masks.utils import apply_masks
from .models.utils import repeat_interleave_batch
from .utils.distributed import AllReduce
from .utils.logging import AverageMeter, CSVLogger, gpu_timer, grad_logger
from .utils.gpu_profiler import create_profiler
# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"  # Better error propagation

_GLOBAL_SEED = 0

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


async def copy_artifacts_wrapper(artifact_location: str, my_artifact_location: str) -> None:
    """
    Wrapper function to copy artifacts with logging.

    Args:
        artifact_location (str): Source artifacts directory
        my_artifact_location (str): Destination artifacts directory
    """
    try:
        logger.info(f"Starting artifact copy from {artifact_location} to {my_artifact_location}")
        await copy_artifacts_async(artifact_location, my_artifact_location)
        logger.info("Artifact copy completed successfully")
    except Exception as e:
        logger.error(f"Failed to copy artifacts: {str(e)}")
        raise

def compute_loss(z: torch.Tensor,
                 h: torch.Tensor,
                 itr: int,
                 world_size: int,
                 world_rank: int,
                 log_freq: int = 100,
                 calc_allreduce: bool = True) -> torch.Tensor:
    """
    Compute the smooth L1 loss between predicted (z) and target (h) embeddings.

    Args:
        z: Predicted embeddings from the predictor network
        h: Target embeddings from the target encoder
        itr: Current iteration number
        world_size: Number of distributed processes
        world_rank: Current process rank
        log_freq: Frequency of logging tensor shapes and memory usage
        calc_allreduce: Whether to perform all-reduce operation for distributed training

    Returns
    -------
        torch.Tensor: Computed loss value
    """
    if world_rank == 0 and itr % log_freq == 0:
        logger.info(f"z shape: {z.shape}, h shape: {h.shape}")
        logger.info(f"z memory: {z.element_size() * z.nelement() / 1024**2:.2f} MB")
        logger.info(f"h memory: {h.element_size() * h.nelement() / 1024**2:.2f} MB")

    loss = F.smooth_l1_loss(z, h, reduction='mean')

    if calc_allreduce:
        # Clone the loss tensor to prevent in-place modifications during all_reduce
        loss = loss.clone()
        # Sum the loss across all distributed processes
        dist.all_reduce(loss, op=dist.ReduceOp.SUM, async_op=True)
        # Average the loss across processes
        loss /= world_size

    return loss


def train(args: dict,
          train_dataset: datasets.Dataset,
          resume_preempt: bool = False,
          save_folder_path: str | None = None,
          my_artifact_location: str | None = None,
          local_rank: int | None = None,
          world_size: int | None = None,
          world_rank: int | None = None,
          ):
    """
    Train model.

    Parameters
    ----------
    args:
        Dictionary containing the hyperparams from the config file.
    train_dataset:
        Train split of huggingface dataset.
    resume_preempt: bool, optional
        Whether to resume from a preempted job.
    save_folder_path: str, optional
        Path for saving model artifacts.
    MY_ARTIFACT_LOCATION: str, optional
        Destination path for copying artifacts.
    LOCAL_RANK: int, optional
        Rank of the process.
    WORLD_SIZE: int, optional
        Total number of processes.
    RANK: int, optional
        Global rank of the process.
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
    elif local_rank is not None:
        device = torch.device(f"cuda:{local_rank}")
    elif local_rank is  None:
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
    count_encoding = args['meta']['count_encoding']
    n_value_bins = args['meta']['n_value_bins']
    enc_depth = args['meta']['enc_depth']
    enc_emb_dim = args['meta']['enc_emb_dim']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if 'num_heads' in args['meta'].keys():
        num_heads = args['meta']['num_heads']
    else:
        num_heads = 8
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

    logger.info(f'Initialized (rank/world-size) {local_rank}/{world_size}.')
    if world_rank > 0:
        logger.setLevel(logging.ERROR)

    # Store config file with model
    if world_rank==0:
        dump = os.path.join(save_folder_path, 'params.yaml')
        with open(dump, 'w') as f:
            yaml.dump(args, f)

    # Define log/checkpointing paths
    # log_file = os.path.join(save_folder_path, f'{write_tag}_r{rank}.csv')
    if world_rank==0:
        save_path = os.path.join(save_folder_path, f'{write_tag}' + '-ep{epoch}.pth.tar')
        latest_path = os.path.join(save_folder_path, f'{write_tag}-latest.pth.tar')
        if load_model:
            load_path = os.path.join(
                load_folder_path, r_file) if r_file is not None else latest_path
    else:
        save_path = None
        latest_path = None
        load_path = None

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
        use_flash_attention=use_flash_attention,
        use_layer_norm=use_layer_norm)
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
    elif cell_masking:
       mask_collator = CelllMaskCollator(
            n_targets=n_targets,
            n_contexts=n_contexts,
            n_segments=n_segments,
            seq_len_cell=seq_len_cell,
            seq_len_neighborhood=seq_len_neighborhood,
            n_special_tokens=n_special_tokens,
            per_block_mask_ratio=per_block_mask_ratio,
            targets_list=targets_list)
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
        rank=local_rank,
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

    encoder = DistributedDataParallel(
        encoder,
        static_graph=True,
        device_ids=[local_rank],
        output_device=local_rank)
    predictor = DistributedDataParallel(
        predictor,
        static_graph=True,
        device_ids=[local_rank],
        output_device=local_rank)
    target_encoder = DistributedDataParallel(
        target_encoder,
        device_ids=[local_rank],
        output_device=local_rank)
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
            scaler=scaler,
            world_rank=world_rank)
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
        if world_rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch) % checkpoint_freq == 0:
                if iter_number is None:
                    torch.save(save_dict, save_path.format(epoch=f'{epoch}'))
                else:
                    torch.save(save_dict, save_path.format(epoch=f'{epoch}_{iter_number}'))

    # Define profiler in a common scope
    profiler = None

    # Initialize profiler at start of epoch
    if world_rank == 0:  # Only profile on main process
        profiler = create_profiler(
            save_folder_path=save_folder_path,
            wait=2,  # Wait 2 steps before profiling
            warmup=2,  # 2 warmup steps
            active=5,  # Profile for 3 steps
            repeat=2,  # Repeat this schedule twice
            use_cuda=True, # Enable CUDA profiling
            with_stack=False, # Disable stack traces
            with_flops=True, # Enable FLOPS counting
            with_modules=True # Enable module-level profiling
        )

    # Run training loop
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}")

        # Update distributed dataloader epoch
        train_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        def train_step(tokens: torch.Tensor,
                       segments: torch.Tensor,
                       positions: torch.Tensor,
                       counts: torch.Tensor,
                       masks_enc: list[torch.Tensor] | None = None,
                       masks_pred: list[torch.Tensor] | None = None,
                       masks_attention: torch.Tensor | None = None,
                       itr: int = 0,
                       epoch: int = 0,
                       profiler: torch.profiler.profile | None = None):
            _new_lr = scheduler.step()
            _new_wd = wd_scheduler.step()

            def forward_target():
                with torch.no_grad():
                    if gt_type == 'rank':
                        h, _, _, _ = target_encoder(
                            tokens=tokens,
                            segments=segments,
                            positions=positions,
                            masks_attention=masks_attention)
                    elif gt_type == 'counts':
                        h, _, _, _ = target_encoder(
                            tokens=tokens,
                            segments=segments,
                            counts=counts,
                            masks_attention=masks_attention)

                    h = F.layer_norm(h, (h.size(-1),))
                    h = apply_masks(h, masks_pred)
                    B = len(h)
                    h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                    return h

            def forward_context():
                if gt_type == 'rank':
                    z, pos_emb, seg_emb, token_emb = encoder(
                        positions=positions,
                        segments=segments,
                        tokens=tokens,
                        masks=masks_enc,
                        masks_attention=None)
                    z = predictor(z=z,
                                pos_embed=pos_emb,
                                segments=segments,
                                token_embed=token_emb,
                                masks_enc=masks_enc,
                                masks_pred=masks_pred,
                                masks_attention=None)
                elif gt_type == 'counts':
                    z, token_emb, seg_emb, value_emb = encoder(
                        tokens=tokens,
                        segments=segments,
                        counts=counts,
                        masks=masks_enc,
                        masks_attention=None)
                    z = predictor(z=z,
                                token_embed=token_emb,
                                segments=segments,
                                counts=counts,
                                masks_enc=masks_enc,
                                masks_pred=masks_pred,
                                masks_attention=None)
                return z

            # Fine-grained profiling for the forward pass and loss computation.
            with torch.profiler.record_function("forward_and_loss"):
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss = compute_loss(z, h, itr, world_size, world_rank, log_freq)

            _enc_norm, _pred_norm = 0., 0.
            if use_bfloat16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            if ((epoch + 1) > warmup) and (clip_grad is not None):
                _enc_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                _pred_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
            if use_bfloat16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if profiler is not None:
                torch.cuda.synchronize()  # Ensure GPU ops complete before stepping profiler
                profiler.step()

            grad_stats = grad_logger(encoder.named_parameters())
            grad_stats.global_norm = float(_enc_norm)
            grad_stats_pred = grad_logger(predictor.named_parameters())
            grad_stats_pred.global_norm = float(_pred_norm)
            optimizer.zero_grad()

            with torch.profiler.record_function("weight_update"):
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters(), strict=True):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

            return (float(loss), _new_lr, _new_wd, grad_stats, grad_stats_pred)

        for itr, (udata, masks_enc, masks_pred, masks_attention) in enumerate(train_loader):
            tokens = udata[0].to(device, non_blocking=True)
            segments = udata[1].to(device, non_blocking=True)
            if gt_type == 'rank':
                positions = udata[2].to(device, non_blocking=True)
                counts = None
            elif gt_type == 'counts':
                counts = udata[2].to(device, non_blocking=True)
                positions = None
            masks_enc = [u.to(device, non_blocking=True) for u in masks_enc]
            masks_pred = [u.to(device, non_blocking=True) for u in masks_pred]
            masks_attention = masks_attention.to(device, non_blocking=True)

            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            if world_rank == 0:  # Only profile on main process
                with profiler:
                    (loss, _new_lr, _new_wd, grad_stats, grad_stats_pred), etime = gpu_timer(
                        lambda: train_step(tokens, segments, positions, counts, masks_enc, masks_pred, masks_attention, itr, epoch, profiler))
                    torch.cuda.synchronize()  # Ensure all asynchronous GPU ops complete
                    profiler.step()
            else:
                (loss, _new_lr, _new_wd, grad_stats, grad_stats_pred), etime = gpu_timer(
                    lambda: train_step(tokens, segments, positions, counts, masks_enc, masks_pred, masks_attention, itr, epoch, profiler))

            loss_meter.update(loss)
            time_meter.update(etime)

            # Logging
            #def log_stats():
            #    csv_logger.log(epoch + 1,
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
            #                    % (epoch + 1, itr,
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
            #                % (epoch + 1, itr,
            #                grad_stats.first_layer,
            #                grad_stats.last_layer,
            #                grad_stats.min,
            #                grad_stats.max))

            #log_stats()
            # if LOCAL_RANK == 0:
            #     wandb.log(
            #         {"loss": loss,
            #         'lr':_new_lr,
            #         'epoch': epoch,
            #         'global_norm_enc': grad_stats.global_norm,
            #         'global_norm_pred': grad_stats_pred.global_norm,
            #         })
            assert not np.isnan(loss), 'loss is nan'
            if itr % checkpoint_freq_iter == 0:
                logger.info(f'Saving checkpoint at epoch {epoch} iteration {itr}')
                save_checkpoint(epoch + 1, itr // checkpoint_freq_iter)

        # -- Save Checkpoint after every epoch
        logger.info(f'avg. loss {loss_meter.avg:.3f}')
        save_checkpoint(epoch + 1)

    logger.info("Training completed")

    # Only attempt to copy artifacts if both paths are provided
    if save_folder_path and my_artifact_location:
        logger.info("Copying artifacts")
        logger.info(f"From: {save_folder_path}")
        logger.info(f"To: {my_artifact_location}")

        # Run the async copy operation
        asyncio.run(copy_artifacts_wrapper(save_folder_path, my_artifact_location))
        logger.info("Artifact copy completed successfully")
    else:
        logger.warning("Skipping artifact copy: save_folder_path or MY_ARTIFACT_LOCATION not provided")
        if not save_folder_path:
            logger.warning("save_folder_path is None")
        if not my_artifact_location:
            logger.warning("my_artifact_location is None")
