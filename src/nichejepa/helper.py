"""
Adapted from Assran, M. et al. Self-supervised learning from images with a
Joint-Embedding Predictive Architecture. Proc. IEEE Comput. Soc. Conf. Comput.
Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/helper.py (05.06.2024).
"""

import logging
import sys
from typing import Literal, Optional, Tuple

import torch

import nichejepa.models.gene_transformers as gt
from .models.utils import trunc_normal_
from .utils.schedulers import (CosineWDSchedule,
                               WarmupCosineSchedule)


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(device: str,
                    r_path: str,
                    encoder: gt.GeneTransformerBaseEncoder,
                    predictor: gt.GeneTransformerBasePredictor,
                    target_encoder: gt.GeneTransformerBaseEncoder,
                    opt: torch.optim.AdamW,
                    scaler: torch.cuda.amp.GradScaler,
                    is_training: bool=True,
                    ) -> Tuple[gt.GeneTransformerBaseEncoder,
                               gt.GeneTransformerBasePredictor,
                               gt.GeneTransformerBaseEncoder,
                               torch.optim.AdamW,
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
    is_training:
        If 'True', load state dict into DDP module.

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

        # Load state into context encoder
        if encoder is not None:
            pretrained_dict = checkpoint['encoder']
            msg = encoder.load_state_dict(pretrained_dict)
            logger.info(
                f'Loaded pretrained encoder from epoch {epoch} with msg: {msg}.')

        # Load state into predictor
        if predictor is not None:
            pretrained_dict = checkpoint['predictor']
            msg = predictor.load_state_dict(pretrained_dict)
            logger.info(
                f'Loaded pretrained predictor from epoch {epoch} with msg: {msg}.')

        # Load state into target encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            if not is_training:
                pretrained_dict = {
                    key.replace("module.", ""): value for key, value in 
                    pretrained_dict.items()}
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(
                f'Loaded pretrained target encoder from epoch {epoch} with msg:'
                f' {msg}.')

        # Load state into optimizer
        if opt is not None:
            opt.load_state_dict(checkpoint['opt'])
            logger.info(f'Loaded optimizer from epoch {epoch}.')
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
            logger.info(f'Loaded scaler from epoch {epoch}.')
        
        logger.info(f'Finished loading checkpoint with read path: {r_path}.')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint: {e}.')
        epoch = 0

    return encoder, predictor, target_encoder, opt, scaler, epoch


def init_model(gt_type: Literal['rank', 'count'],
               n_value_bins: int,
               device: str,
               vocab_size: int,
               seq_len: int,
               n_special_tokens: int,
               n_segments: int,
               n_special_values: Optional[int]=None,
               enc_emb_dim: int=768, 
               enc_depth: int=12,
               pred_emb_dim: int=384,
               pred_depth: int=6,
               use_flash_attention: bool=True,
               ) -> Tuple[gt.GeneTransformerBaseEncoder,
                          gt.GeneTransformerBasePredictor]:
    """
    Initialize model.

    Parameters
    -----------
    gt_type:
        Gene transformer type.
    n_value_bins:
        Number of value bins for count tokenization.s
    device:
        Device on which the model will be initialized.
    vocab_size:
        Size of the token vocabulary. Includes <pad> token.
    seq_len:
        Length of the token sequences (w/o <cls> token).
    n_special_tokens:
    n_segments:
    n_special_values:
        Number of special values.
    enc_emb_dim:
        Dimension of the encoder embedding.
    enc_depth:
        Number of transformer blocks in the encoder.
    pred_emb_dim:
        Dimension of the predictor embedding.        
    pred_depth:
        Number of transformer blocks in the predictor.
    use_flash_attention:
        If use flash_attention or not
    Returns
    -----------
    encoder:
        Initialized GeneTransformerEncoder module.
    predictor:
        Initialized GeneTransformerPredictor module.
    """
    encoder = gt.__dict__["init_gt_encoder"](
        encoder_type=gt_type,
        n_value_bins=n_value_bins,
        n_special_values=n_special_values,
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_special_tokens=n_special_tokens,
        n_segments=n_segments,
        embed_dim=enc_emb_dim,
        depth=enc_depth,
        use_flash_attention=use_flash_attention)
    predictor = gt.__dict__["init_gt_predictor"](
        predictor_type=gt_type,
        embed_dim=enc_emb_dim,
        seq_len=seq_len,
        n_special_tokens=n_special_tokens,
        n_segments=n_segments,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        use_flash_attention=use_flash_attention)

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

    
def init_opt(encoder: gt.GeneTransformerBaseEncoder,
             predictor: gt.GeneTransformerBasePredictor,
             iterations_per_epoch: int,
             start_lr: float,
             ref_lr: float,
             warmup: int,
             num_epochs: int,
             wd: float=1e-6,
             final_wd: float=1e-6,
             final_lr: float=0.0,
             use_bfloat16: bool=False,
             ipe_scale: float=1.25
             ) -> Tuple[torch.optim.AdamW,
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
