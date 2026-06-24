"""
Adapted from Assran, M. et al. Self-supervised learning from images with
a Joint-Embedding Predictive Architecture. Proc. IEEE Comput. Soc. Conf.
Comput. Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/helper.py
(05.06.2024).
"""

import logging
import math
from typing import Literal

import torch
#from peft import get_peft_model, LoraConfig

import terra.models.gene_transformers as gt
from terra.models.batch_classifier import BatchClassifierHead
from terra.models.multimask import (EncoderMultiMaskWrapper,
                                        PredictorMultiMaskWrapper)
from terra.models.utils import trunc_normal_
from terra.utils.schedulers import (CosineWDSchedule,
                                        WarmupCosineSchedule)


logger = logging.getLogger(__name__)

"""
def apply_peft(model, peft_method='lora', rank=8):
    if peft_method == 'lora':
        peft_config = LoraConfig(r=rank)
        model = get_peft_model(model, peft_config)
    return model
"""

def parse_distribution_alignment_kwargs(args: dict) -> dict | None:
    """Resolve the optional
    ``special_token_correction.distribution_alignment`` config block
    (or the legacy alias ``batch_correction.distribution_alignment``).
    Returns ``None`` when missing or disabled so the training loop
    can gate the alignment loss on a simple None check.

    Returned dict also propagates any per-mechanism multi-key spec
    (``keys`` / ``n_classes`` / ``offsets``) so the runtime loop in
    train.py can route distribution_alignment by one or more
    metadata fields.

    Expected YAML schema (any field can be omitted, defaults shown):

    .. code-block:: yaml

        special_token_correction:
          distribution_alignment:
            enabled:         True
            method:          'coral'           # 'coral', 'mmd', or 'sinkhorn'
            lambda:          0.1               # loss weight
            mmd_sigmas:      [0.1, 1.0, 10.0]  # only used if method='mmd'
            sinkhorn_eps:    0.05              # only used if method='sinkhorn'
            sinkhorn_n_iter: 100               # only used if method='sinkhorn'
            max_pairs:       null              # cap on pair count per step
            # Optional per-mechanism multi-key override (otherwise
            # inherited from the top-level spec):
            keys:            ['batch_value']
            offsets:         [2]
    """
    # Accept either the new block name or the legacy alias.
    bc = (args.get('special_token_correction', None)
          or args.get('batch_correction', None)
          or {})
    cfg = bc.get('distribution_alignment', None)
    if not (cfg and cfg.get('enabled', False)):
        return None
    method = str(cfg.get('method', 'coral')).lower()
    if method not in ('coral', 'mmd', 'sinkhorn'):
        raise ValueError(
            "distribution_alignment.method must be "
            f"'coral', 'mmd', or 'sinkhorn' (got {method!r}).")
    out = {
        'method':          method,
        'lambda':          float(cfg.get('lambda', 0.1)),
        'mmd_sigmas':      list(cfg.get('mmd_sigmas', [0.1, 1.0, 10.0])),
        'sinkhorn_eps':    float(cfg.get('sinkhorn_eps', 0.05)),
        'sinkhorn_n_iter': int(cfg.get('sinkhorn_n_iter', 100)),
        'max_pairs':       cfg.get('max_pairs', None),
    }
    if out['max_pairs'] is not None:
        out['max_pairs'] = int(out['max_pairs'])

    # Propagate multi-key spec (keys/offsets) so train.py can route
    # distribution_alignment by one or more metadata fields. If the
    # per-mechanism cfg doesn't set them, inherit from the top-level
    # ``special_token_correction`` block.
    from terra.models.batch_labels import resolve_label_spec
    spec = resolve_label_spec(cfg, shared={
        'keys': bc.get('keys'),
        'n_classes': bc.get('n_classes'),
        'offsets': bc.get('offsets'),
    })
    if spec['keys']:
        out['keys'] = spec['keys']
        out['offsets'] = spec['offsets']
        # n_classes is not needed for dist_align (it auto-detects
        # unique values per minibatch) but pass it through for
        # consistency / round-trip.
        out['n_classes'] = spec['n_classes']
    return out


def parse_arch_kwargs(args: dict) -> dict:
    """Parse architecture hyperparameters that must round-trip from a
    saved ``params.yaml`` into ``init_model`` when rebuilding a
    trained encoder for fine-tuning or inference.

    Returns a dict suitable for direct ``**unpacking`` into
    ``init_model``. Covers: Laplacian PE (laplacian_k, laplacian_sigma),
    RoPE (rope_freq_scale, rope_rotation_augment), and AdaLN
    (adaln_kwargs). For any field missing from the config, the
    corresponding default in ``init_model``'s signature applies.

    Use this in every entry point that reconstructs a trained
    encoder, so a checkpoint trained with non-default hyperparameters
    rebuilds with the same shapes / dynamics. Skipping it means
    state_dict load may silently succeed but the runtime behavior
    differs from training (e.g. wrong Laplacian sigma -> wrong
    adjacency -> wrong embeddings).
    """
    meta = args.get('meta', {}) or {}
    # Accept both 'special_token_correction' (new, recommended) and
    # 'batch_correction' (legacy alias) as the block name.
    bc = (args.get('special_token_correction', None)
          or args.get('batch_correction', None)
          or {})
    adaln = bc.get('adaln', None)
    if adaln and not adaln.get('enabled', False):
        adaln = None
    elif adaln is not None:
        from terra.models.batch_labels import resolve_label_spec
        adaln = dict(adaln)
        # Resolve the multi-key spec for round-trip. Falls back to
        # shared top-level keys/n_classes/offsets if the per-
        # mechanism block doesn't have them. If still empty, fall
        # back to the legacy single-key ``n_batches`` field.
        spec = resolve_label_spec(adaln, shared={
            'keys': bc.get('keys'),
            'n_classes': bc.get('n_classes'),
            'offsets': bc.get('offsets'),
        })
        if spec['keys']:
            adaln.setdefault('keys', spec['keys'])
            adaln.setdefault('n_classes', spec['n_classes'])
            adaln.setdefault('offsets', spec['offsets'])
        else:
            adaln.setdefault('n_batches', bc.get('n_batches'))
            if adaln.get('n_batches') is None:
                raise ValueError(
                    "adaln.enabled=True in the saved config but "
                    "neither a multi-key spec (keys/n_classes/"
                    "offsets) nor the legacy n_batches scalar is "
                    "set. Either set keys + n_classes + offsets "
                    "(at the adaln or top-level block), or set "
                    "n_batches.")
        # Round-trip 'scope' as-is so reloaded checkpoints rebuild
        # the same encoder/predictor module shapes. Validate early
        # so a typo blows up at init time rather than at first
        # forward pass.
        scope = str(adaln.get('scope', 'both')).lower()
        if scope not in ('encoder', 'predictor', 'both'):
            raise ValueError(
                "batch_correction.adaln.scope must be one of "
                f"'encoder', 'predictor', 'both' (got {scope!r}).")
        adaln['scope'] = scope
    # Read-Depth-Aware (RDA) conditioning. Round-trips as-is so a
    # checkpoint trained with rda.enabled=True rebuilds the
    # encoder with the matching depth-conditioning hypernetwork.
    rda = args.get('rda', None)
    if rda and not rda.get('enabled', False):
        rda = None
    elif rda is not None:
        rda = dict(rda)

    # Special-token-MoE predictor bias. Reads
    # ``batch_correction.special_token_moe`` (or the legacy alias
    # ``batch_correction.protocol_moe`` for backward compat).
    # Round-trips so the predictor's per-slot Embedding tables are
    # recreated with the same n_experts at checkpoint reload. The
    # multi-slot schema accepts lists for routing_indices,
    # n_experts, and routing_offsets; single-slot scalars are also
    # accepted (and treated as one-element lists internally).
    moe = bc.get('special_token_moe', None)
    if moe is None:
        moe = bc.get('protocol_moe', None)  # legacy alias
    if moe and not moe.get('enabled', False):
        moe = None
    elif moe is not None:
        moe = dict(moe)
        # Resolve multi-key spec with fallback to the shared
        # top-level keys/n_classes/offsets. This way a MoE config
        # that inherits from ``special_token_correction.keys``
        # round-trips correctly at inference time too.
        from terra.models.batch_labels import resolve_label_spec
        spec = resolve_label_spec(moe, shared={
            'keys': bc.get('keys'),
            'n_classes': bc.get('n_classes'),
            'offsets': bc.get('offsets'),
        })
        if spec['keys']:
            # Translate from the unified spec to MoE's internal API
            # names (routing_keys / n_experts / routing_offsets).
            moe.setdefault('routing_keys', spec['keys'])
            moe.setdefault('n_experts', spec['n_classes'])
            moe.setdefault('routing_offsets', spec['offsets'])
        has_routing = any(
            moe.get(name) is not None for name in (
                'routing_keys', 'routing_key',
                'routing_indices', 'routing_index',
            )
        )
        if not has_routing:
            raise ValueError(
                "special_token_moe.enabled=True in the saved config "
                "but no routing spec found. The MoE needs one of:\n"
                "  - special_token_correction.keys (shared), or\n"
                "  - special_token_moe.routing_keys, or\n"
                "  - special_token_moe.routing_indices (legacy).")
        if moe.get('n_experts') is None:
            raise ValueError(
                "special_token_moe.enabled=True in the saved config "
                "but n_experts is missing. Set it (scalar or list) "
                "to at least 1 + the max routing-id observed.")
    return {
        'laplacian_k': int(meta.get('laplacian_k', 8)),
        'laplacian_sigma': float(meta.get('laplacian_sigma', 1.0)),
        'rope_freq_scale': meta.get('rope_freq_scale', None),
        'rope_rotation_augment': bool(
            meta.get('rope_rotation_augment', True)),
        'adaln_kwargs': adaln,
        'rda_kwargs': rda,
        'special_token_moe_kwargs': moe,
    }


def parse_protein_init_kwargs(args: dict,
                              token_dict: dict | None = None,
                              ) -> dict | None:
    """Resolve the optional ``protein_init`` config block into encoder kwargs.

    Mirrors the parsing used by ``train.py`` so other entry points
    (``finetune.py``, every script under ``terra/inference``) can rebuild
    a protein-init-trained encoder with one call.

    Returns ``None`` if the ``protein_init`` section is absent or
    ``enabled: false``, in which case the encoder falls back to the
    default learnable ``nn.Embedding`` (existing behavior).
    """
    cfg = args.get('protein_init', None)
    if not (cfg and cfg.get('enabled', False)):
        logger.info(
            "Protein-init: DISABLED -- using the default learnable "
            "nn.Embedding for gene tokens.")
        return None
    kwargs = {
        'embedding_path':  cfg['embedding_path'],
        'mapping_path':    cfg['mapping_path'],
        'mode':            cfg.get('mode', 'routing'),
        'proj_bias':       cfg.get('proj_bias', False),
        'use_layer_norm':  cfg.get('use_layer_norm', True),
        'freeze_esm':      cfg.get('freeze_esm', True),
        'warm_start_target_std': cfg.get('warm_start_target_std', 1.0),
        'warm_start_seed':       cfg.get('warm_start_seed', 0),
    }
    # Optional: override which Ensembl gene-ID prefixes count as gene
    # tokens (default in protein_init covers human ENSG + mouse ENSMUSG).
    # YAML can pass a single string or a list.
    if 'gene_id_prefixes' in cfg:
        prefixes = cfg['gene_id_prefixes']
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        kwargs['gene_id_prefixes'] = list(prefixes)
    if kwargs['mode'] == 'warm_start':
        mode_desc = (
            "warm-start (PCA-reduced ESM into plain nn.Embedding, "
            "architecture identical to baseline)"
        )
    else:
        mode_desc = (
            "routing (frozen ESM + learnable projection, UCE-style)"
            if kwargs['freeze_esm']
            else "routing (ESM as init, matrix TRAINABLE, no weight decay)"
        )
    logger.info(
        "Protein-init: ENABLED -- %s. embedding=%s | mapping=%s | "
        "mode=%s%s",
        mode_desc,
        kwargs['embedding_path'],
        kwargs['mapping_path'],
        kwargs['mode'],
        f" | gene_id_prefixes={kwargs['gene_id_prefixes']}"
        if 'gene_id_prefixes' in kwargs else "",
    )
    if token_dict is not None:
        kwargs['token_dict'] = token_dict
    return kwargs


def load_checkpoint(device: str,
                    r_path: str,
                    encoder: gt.GeneTransformerBaseEncoder,
                    predictor: gt.GeneTransformerBasePredictor,
                    target_encoder: gt.GeneTransformerBaseEncoder,
                    opt: torch.optim.AdamW,
                    scaler: torch.cuda.amp.GradScaler,
                    is_training: bool = True,
                    ) -> tuple[gt.GeneTransformerBaseEncoder,
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
        Initialized GeneTransformerPredictor module to predict targets
        from contexts.
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
        GeneTransformerEncoder module to encode contexts, loaded with
        state from checkpoint.
    predictor:
        GeneTransformerPredictor module to predict targets from
        contexts, loaded with state from checkpoint.
    target_encoder:
        GeneTransformerEncoder module to encode targets, loaded with
        state from checkpoint.
    opt:
        Torch optimizer, loaded with stete from checkpoint.
    scaler:
        Torch scaler for automatic mixed precision training, loaded with
        state from checkpoint.
    epoch:
        Number of epochs from checkpoint.
    """
    try:
        checkpoint = torch.load(
            r_path, map_location=torch.device(device))
        
        if ('zero_epoch_tracking' in checkpoint.keys()) and not ('iter_number' in checkpoint.keys()):
            epoch = checkpoint['epoch'] + 1
        else: # just for backwards compatibility
            epoch = checkpoint['epoch']

        # TO DO: Update
        if 'iter_number' in checkpoint.keys():
            iter_number = checkpoint['iter_number']
        else:
            iter_number = None

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
        iter_number = None

    return encoder, predictor, target_encoder, opt, scaler, epoch, iter_number


def init_model(gt_type: Literal['rank', 'count', 'combined'],
               count_encoding: Literal['value_bins', 'mlp'],
               n_value_bins: int,
               cell_pos_enc: Literal[
                   'none', 'segment', 'coord', 'polar', 'alibi',
                   'polar+alibi', 'laplacian', 'rope'],
               device: str,
               vocab_size: int,
               seq_len: int,
               n_special_tokens: int,
               n_segments: int,
               n_special_values: int | None = None,
               enc_emb_dim: int = 768, 
               enc_depth: int = 12,
               pred_emb_dim: int = 384,
               pred_depth: int = 6,
               num_heads: int = 8,
               mlp_ratio: float = 4.0,
               use_flash_attention: bool = True,
               use_layer_norm: bool = True,
               api_version: Literal['v1', 'v2', 'v3'] = 'v4',
               sep_gene_tokens_neb: bool = False,
               predict_gene: bool = True,
               pos_learnable: bool = False,
               nz_spc: bool = False,
               new_spc: bool = False,
               mlp_bias: bool = False,
               protein_init_kwargs: dict | None = None,
               laplacian_k: int = 8,
               laplacian_sigma: float = 1.0,
               rope_freq_scale: float | None = None,
               rope_rotation_augment: bool = True,
               adaln_kwargs: dict | None = None,
               rda_kwargs: dict | None = None,
               special_token_moe_kwargs: dict | None = None,
               ) -> tuple[gt.GeneTransformerBaseEncoder,
                          gt.GeneTransformerBasePredictor]:
    """
    Initialize model.

    Parameters
    -----------
    gt_type:
        Gene transformer type.
    count_encoding:
        How counts are encoded.
    n_value_bins:
        Number of value bins if `value_bin` count encoding is used.
    cell_pos_enc:
        How cell positions are encoded.
    device:
        Device on which the model will be initialized.
    vocab_size:
        Size of the token vocabulary. Includes <pad> token.
    seq_len:
        Length of the token sequences (w/o <cls> token).
    n_special_tokens:
        Number of special tokens.
    n_segments:
        Number of segments.
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
        If `True` use flash_attention.
    use_layer_norm:
        If `True` use layer norm, else Dynamic Tanh.
    sep_gene_tokens_neb:
        If `True`, use separate gene tokens for neighborhood.
    predict_gene:
        If `True`, predict gene given rank, otherwise predict rank given
        gene.

    Returns
    -----------
    encoder:
        Initialized GeneTransformerEncoder module.
    predictor:
        Initialized GeneTransformerPredictor module.
    """
    # AdaLN scope: choose whether the conditioning hypernetwork is
    # applied in the encoder, the predictor, or both. Default 'both'
    # preserves the original behavior; 'predictor' follows the
    # scGPT-spatial / LLOKI pattern (encoder learns batch-invariant
    # representations; only the decoder/predictor uses batch
    # conditioning).
    adaln_kwargs_enc = None
    adaln_kwargs_pred = None
    if adaln_kwargs is not None:
        adaln_inner = dict(adaln_kwargs)
        scope = str(adaln_inner.pop('scope', 'both')).lower()
        if scope not in ('encoder', 'predictor', 'both'):
            raise ValueError(
                "batch_correction.adaln.scope must be one of "
                f"'encoder', 'predictor', 'both' (got {scope!r}).")
        if scope in ('encoder', 'both'):
            adaln_kwargs_enc = dict(adaln_inner)
        if scope in ('predictor', 'both'):
            adaln_kwargs_pred = dict(adaln_inner)
        logger.info(
            f"[AdaLN] scope='{scope}' "
            f"(encoder: {'on' if adaln_kwargs_enc else 'off'}, "
            f"predictor: {'on' if adaln_kwargs_pred else 'off'})")

    # RDA depth conditioning is encoder-only (predictor doesn't see
    # raw counts). Banner mirrors AdaLN's style.
    if rda_kwargs and rda_kwargs.get('enabled', False):
        logger.info(
            f"[RDA] enabled (hidden_dim="
            f"{rda_kwargs.get('hidden_dim', 32)}, "
            f"use_target_depth="
            f"{bool(rda_kwargs.get('use_target_depth', False))}). "
            "Per-cell log(1+T) is broadcast across gene-token "
            "positions; zero-init output head -> step-0 no-op.")

    # Special-token MoE predictor bias is predictor-only (encoder
    # remains protocol/batch-agnostic). Zero-init biases -> step-0
    # no-op. Multi-slot routing sums per-slot biases (e.g. batch +
    # assay contributes both per-batch and per-assay biases summed).
    if special_token_moe_kwargs and special_token_moe_kwargs.get(
            'enabled', False):
        ri = (special_token_moe_kwargs.get('routing_indices')
              if special_token_moe_kwargs.get('routing_indices') is not None
              else special_token_moe_kwargs.get('routing_index'))
        ne = special_token_moe_kwargs.get('n_experts')
        ro = (special_token_moe_kwargs.get('routing_offsets')
              if special_token_moe_kwargs.get('routing_offsets') is not None
              else special_token_moe_kwargs.get('routing_offset', 0))
        logger.info(
            f"[Special-Token-MoE] enabled (routing_indices={ri}, "
            f"n_experts={ne}, routing_offsets={ro}). Per-slot "
            "additive bias on predictor output (summed across "
            "slots); zero-init -> step-0 no-op. NOTE: each "
            "values[:, routing_index] is offset-subtracted across "
            "ALL spv_* slots; set routing_offset per slot to the "
            "cumulative size of preceding slots, or set n_experts "
            "large enough to cover the full global range.")

    encoder = gt.__dict__["init_gt_encoder"](
        encoder_type=gt_type,
        n_special_values=n_special_values,
        count_encoding=count_encoding,
        n_value_bins=n_value_bins,
        cell_pos_enc=cell_pos_enc,
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_special_tokens=n_special_tokens,
        n_segments=n_segments,
        embed_dim=enc_emb_dim,
        depth=enc_depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        use_flash_attention=use_flash_attention,
        use_layer_norm=use_layer_norm,
        api_version=api_version,
        sep_gene_tokens_neb=sep_gene_tokens_neb,
        pos_learnable=pos_learnable,
        nz_spc=nz_spc,
        mlp_bias=mlp_bias,
        protein_init_kwargs=protein_init_kwargs,
        laplacian_k=laplacian_k,
        laplacian_sigma=laplacian_sigma,
        rope_freq_scale=(rope_freq_scale if rope_freq_scale is not None
                         else math.pi),
        rope_rotation_augment=rope_rotation_augment,
        adaln_kwargs=adaln_kwargs_enc,
        rda_kwargs=rda_kwargs)
    if api_version == 'v3' or api_version == 'v4':
        encoder = EncoderMultiMaskWrapper(encoder)
    predictor = gt.__dict__["init_gt_predictor"](
        predictor_type=gt_type,
        n_special_values=n_special_values,
        embed_dim=enc_emb_dim,
        seq_len=seq_len,
        n_special_tokens=n_special_tokens,
        n_segments=n_segments,
        cell_pos_enc=cell_pos_enc,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        use_flash_attention=use_flash_attention,
        use_layer_norm=use_layer_norm,
        api_version=api_version,
        predict_gene=predict_gene,
        pos_learnable=pos_learnable,
        nz_spc=nz_spc,
        new_spc=new_spc,
        rope_freq_scale=(rope_freq_scale if rope_freq_scale is not None
                         else math.pi),
        rope_rotation_augment=rope_rotation_augment,
        adaln_kwargs=adaln_kwargs_pred,
        special_token_moe_kwargs=special_token_moe_kwargs)
    if api_version == 'v3' or api_version == 'v4':
        predictor = PredictorMultiMaskWrapper(predictor)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            # Affine-less LayerNorm (e.g. inside AdaLN) has
            # m.bias / m.weight = None. Guard against that case so the
            # post-construction re-init pass doesn't crash when AdaLN
            # is enabled.
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    # The init_weights loop above overwrites AdaLN's modulation
    # hypernetwork with trunc_normal_(0.02). Restore the zero-init so
    # the AdaLN-at-step-0 == LayerNorm invariant survives this second
    # reinit pass. No-op when AdaLN is disabled.
    from terra.models.adaln import zero_init_adaln_modulations
    zero_init_adaln_modulations(encoder)
    zero_init_adaln_modulations(predictor)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f'Encoder number of parameters: {count_parameters(encoder)}')
    logger.info(
        f'Predictor number of parameters: {count_parameters(predictor)}')

    return encoder, predictor


def build_batch_classifier_head(
        adv_classifier_kwargs: dict | None,
        embed_dim: int,
        device: str,
        ) -> torch.nn.Module | None:
    """Construct the adversarial batch classifier head.

    Returns None when ``adv_classifier_kwargs`` is missing / disabled,
    so the existing two-tuple return of ``init_model`` and the inference
    scripts are unaffected. When enabled, returns a
    ``BatchClassifierHead`` on ``device``.

    Parameters
    ----------
    adv_classifier_kwargs:
        Config dict. Required keys when ``enabled=True``:
        ``n_batches`` (int, max batch ID + 1 in the corpus).
        Optional keys: ``hidden_dim`` (default 256), ``dropout``
        (default 0.1).
    embed_dim:
        Cell embedding dimension at the input (= encoder ``enc_emb_dim``).
    device:
        Torch device.
    """
    if not (adv_classifier_kwargs
            and adv_classifier_kwargs.get('enabled', False)):
        return None
    from terra.models.batch_labels import resolve_label_spec
    spec = resolve_label_spec(adv_classifier_kwargs)
    if not spec['n_classes']:
        # Legacy single-key without an explicit ``keys`` field.
        n_classes_per_key = [
            int(adv_classifier_kwargs['n_batches'])]
    else:
        n_classes_per_key = spec['n_classes']
    head = BatchClassifierHead(
        embed_dim=embed_dim,
        n_classes_per_key=n_classes_per_key,
        hidden_dim=int(adv_classifier_kwargs.get('hidden_dim', 256)),
        dropout=float(adv_classifier_kwargs.get('dropout', 0.1)),
    ).to(device)

    def _init_w(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    for m in head.modules():
        _init_w(m)

    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    logger.info(
        f'Adversarial batch classifier head: {n_params} params, '
        f'n_classes_per_key={n_classes_per_key}.')
    return head

    
def init_opt(encoder: gt.GeneTransformerBaseEncoder,
             predictor: gt.GeneTransformerBasePredictor,
             iterations_per_epoch: int,
             start_lr: float,
             ref_lr: float,
             warmup: int,
             num_epochs: int,
             wd: float = 1e-6,
             final_wd: float = 1e-6,
             final_lr: float = 0.0,
             use_bfloat16: bool = False,
             ipe_scale: float = 1.25,
             api_version: Literal['v1', 'v2', 'v3', 'v4'] = 'v4',
             ) -> tuple[torch.optim.AdamW,
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
    api_version:

    Returns
    -----------
    optimizer:
    scaler:
    scheduler:
    wd_scheduler:    
    """
    # A param goes into the no-weight-decay group if it's a bias, a 1-D
    # weight (LayerNorm scale, etc.), OR has been explicitly flagged
    # by a module setting ``param._no_weight_decay = True`` on it.
    # The flag is used by protein-init's unfrozen ESM matrix so weight
    # decay can't erode the pretrained prior over training.
    def _no_wd(name, param) -> bool:
        if 'bias' in name or len(param.shape) == 1:
            return True
        if getattr(param, '_no_weight_decay', False):
            return True
        return False

    param_groups = [{'params': [p for n, p in encoder.named_parameters()
                                if not _no_wd(n, p)]},
                    {'params': [p for n, p in predictor.named_parameters()
                                if not _no_wd(n, p)]},
                    {'params': [p for n, p in encoder.named_parameters()
                                if _no_wd(n, p)],
                     'WD_exclude': True,
                     'weight_decay': 0},
                    {'params': [p for n, p in predictor.named_parameters()
                                if _no_wd(n, p)],
                     'WD_exclude': True,
                     'weight_decay': 0}]

    # Initialize optimizer with decoupled weight decay
    logger.info('Initializing optimizer: AdamW.')
    optimizer = torch.optim.AdamW(param_groups, fused=True)

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
    #scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    scaler = None # GradScaler should not be used for bfloat16
    
    return optimizer, scaler, scheduler, wd_scheduler