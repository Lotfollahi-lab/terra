"""
Adapted from Assran, M. et al. Self-supervised learning from images with a
Joint-Embedding Predictive Architecture. Proc. IEEE Comput. Soc. Conf. Comput.
Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/helper.py (05.06.2024).
"""

import logging
import sys
from typing import Tuple

import torch

import nichejepa.models.gene_transformer as gt
from .utils.schedulers import (WarmupCosineSchedule,
                               CosineWDSchedule)
from .utils.tensors import trunc_normal_


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(
    device: str,
    r_path: str,
    encoder: gt.GeneTransformerEncoder,
    predictor: gt.GeneTransformerPredictor,
    target_encoder: gt.GeneTransformerEncoder,
    opt: torch.optim.adamw.AdamW,
    scaler: torch.cuda.amp.GradScaler) -> Tuple[gt.GeneTransformerEncoder,
                                                gt.GeneTransformerPredictor,
                                                gt.GeneTransformerEncoder,
                                                torch.optim.adamw.AdamW,
                                                torch.cuda.amp.GradScaler,
                                                int]:
    """
    Load model checkpoint from stored file.

    Parameters
    -----------
    device:
        Device where the checkpoint will be loaded to.
    r_path:
        Path to the stored checkpoint to be loaded.
    encoder:
        Initialized GeneTransformerEncoder module to encode contexts.
    predictor:
        Initialized GeneTransformerPredictor module to predict targets from
        contexts.
    target_encoder:
        Initialized GeneTransformerEncoder module to encode targets.
    opt:
        Torch optimizer.
    scaler:
        Torch scaler for automatic mixed precision training.

    Returns
    -----------
    encoder:
        GeneTransformerEncoder module to encode contexts, loaded with state from
        checkpoint.
    predictor:
        GeneTransformerPredictor module to predict targets from contexts, loaded
        with state from checkpoint.
    target_encoder:
        GeneTransformerEncoder module to encode targets, loaded with state from
        checkpoint.
    opt:
        Torch optimizer, loaded with stete from checkpoint.
    scaler:
        Torch scaler for automatic mixed precision training, loaded with state
        from checkpoint.
    epoch:
        Number of epochs from checkpoint.
    """
    try:
        checkpoint = torch.load(r_path, map_location=torch.device(device))

        epoch = checkpoint['epoch']

        # Load encoder
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(
            f'Loaded pretrained encoder from epoch {epoch} with msg: {msg}.')

        # Load predictor
        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(
            f'loaded pretrained predictor from epoch {epoch} with msg: {msg}.')

        # Load target encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(
                f'Loaded pretrained target encoder from epoch {epoch} with msg:'
                f' {msg}.')

        # Load optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'Loaded optimizer from epoch {epoch}.')
        logger.info(f'Finished loading checkpoint with read path: {r_path}.')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint: {e}.')
        epoch = 0

    return encoder, predictor, target_encoder, opt, scaler, epoch


def init_model(device: str,
               seq_len: int,
               enc_emb_dim: int=192, 
               enc_depth: int=12, 
               model_name: str="gt_base",
               pred_depth: int=6,
               pos_learnable=1,
               vocab_size: int=6033,
               pred_emb_dim: int=384,
               has_cls: bool=True) -> Tuple[gt.GeneTransformerEncoder,
                                            gt.GeneTransformerPredictor]:
    """
    Initialize model.

    Parameters
    -----------
    device:
    seq_len:
    enc_emb_dim:
    enc_depth:
    model_name:
    pred_depth:
    pos_learnable:
    vocab_size:
    pred_emb_dim:
    has_cls:

    Returns
    -----------  
    """
    encoder = gt.__dict__[model_name](
        seq_len=seq_len,
        embed_dim=enc_emb_dim,
        pos_learnable=pos_learnable,
        depth=enc_depth,
        vocab_size=vocab_size,
        has_cls=True)
    predictor = gt.__dict__["gt_predictor"](
        seq_len=seq_len,
        embed_dim=encoder.embed_dim,
        pos_learnable=pos_learnable,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
        has_cls=True)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    
    return encoder, predictor

    
def init_opt(encoder: gt.GeneTransformerEncoder,
             predictor: gt.GeneTransformerPredictor,
             iterations_per_epoch: int,
             start_lr: float,
             ref_lr: float,
             warmup: int,
             num_epochs: int,
             wd: float=1e-6,
             final_wd: float=1e-6,
             final_lr: float=0.0,
             use_bfloat16: bool=False,
             ipe_scale: float=1.25) -> Tuple[torch.optim.adamw.AdamW,
                                             torch.cuda.amp.GradScaler,
                                             WarmupCosineSchedule,
                                             CosineWDSchedule]:
    """
    Initialize optimizer, learning rate scheduler, weight decay scheduler, and
    automatic mixed precision scaler.

    Parameters
    -----------
    encoder:
    predictor:
    iterations_per_epoch:
    start_lr:
    ref_lr:
    warmup:
    num_epochs:
    wd:
    final_wd:
    final_lr:
    use_bfloat16:
    ipe_scale:

    Returns
    -----------
    optimizer:
    scaler:
    scheduler:
    wd_scheduler:    
    """
    param_groups = [{'params': (p for n, p in encoder.named_parameters()
                                if ('bias' not in n) and (len(p.shape) != 1))},
                    {'params': (p for n, p in predictor.named_parameters()
                                if ('bias' not in n) and (len(p.shape) != 1))},
                    {'params': (p for n, p in encoder.named_parameters()
                                if ('bias' in n) or (len(p.shape) == 1)),
                     'WD_exclude': True,
                     'weight_decay': 0},
                    {'params': (p for n, p in predictor.named_parameters()
                                if ('bias' in n) or (len(p.shape) == 1)),
                     'WD_exclude': True,
                     'weight_decay': 0}]

    # Initialize optimizer with decoupled weight decay
    logger.info('Initializing optimizer: AdamW.')
    optimizer = torch.optim.AdamW(param_groups)

    # Initialize learning rate scheduler
    logger.info('Initializing learning rate scheduler: WarmupCosineSchedule.')
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    
    # Initialize weight decay scheduler
    logger.info('Initializing weight decay scheduler: CosineWDSchedule.')
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    
    # Initialize gradient scaler for automatic mixed precision training to
    # increase the loss magnitude, ensuring gradients are large enough to be
    # represented in FP16
    logger.info('Initializing automatic mixed precision training scaler: '
                'GradScaler.')
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    
    return optimizer, scaler, scheduler, wd_scheduler
