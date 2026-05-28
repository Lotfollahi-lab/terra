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

from app.utils import (build_batch_classifier_head,
                       init_model, init_opt, load_checkpoint,
                       parse_distribution_alignment_kwargs,
                       parse_protein_init_kwargs)
from nichejepa.models.batch_classifier import (
    grad_reverse, mean_pool_cell_embedding)
from nichejepa.models.cycle_consistency import (
    cycle_consistency_loss, make_swapped_batch)
from nichejepa.models.distribution_alignment import (
    compute_distribution_alignment_loss)
from nichejepa.datasets.cell_datasets import init_cell_dataset
from nichejepa.datasets.dataloaders import init_dataloader_and_sampler
from nichejepa.masks.block_masking  import BlockMaskCollator
from nichejepa.masks.cell_masking import CellMaskCollator
from nichejepa.masks.utils import apply_masks
from nichejepa.models.utils import repeat_interleave_batch
from nichejepa.utils.distributed import init_distributed


def _all_gather_with_local_grad(
        local: torch.Tensor) -> torch.Tensor:
    """All-gather a per-rank tensor across DDP ranks.

    Gradients flow through THIS rank's contribution only -- other
    ranks' slots are detached values produced by ``dist.all_gather``
    (which has no autograd). This is the standard pattern used by
    MoCo / SimCLR / VICReg / Barlow Twins distributed training:
    each rank computes the loss on the concatenated pool from all
    ranks (so the cov / var statistics see a large effective sample
    size), but backprop only updates the local subset's parameters,
    which is exactly what DDP's all-reduce of gradients then
    averages across ranks.

    Returns ``local`` unchanged when distributed is not initialized
    or world_size == 1.
    """
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        return local
    ws = dist.get_world_size()
    if ws == 1:
        return local
    rank = dist.get_rank()
    gathered = [torch.zeros_like(local) for _ in range(ws)]
    dist.all_gather(gathered, local.contiguous())
    # Restore the gradient-tracked local tensor in this rank's slot
    # so backward() can flow through the loss term.
    gathered[rank] = local
    return torch.cat(gathered, dim=0)
from nichejepa.utils.logging import (AverageMeter,
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
    if 'mega_batch_mult_max' in args['data'].keys():
        mega_batch_mult_max = args['data']['mega_batch_mult_max']
    else:
        mega_batch_mult_max = 1000

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
    if 'new_spc' in args['meta'].keys():
        new_spc = args['meta']['new_spc']
    else:
        new_spc = False
    if 'excl_spc_from_loss' in args['meta'].keys():
        excl_spc_from_loss = args['meta']['excl_spc_from_loss']
    else:
        excl_spc_from_loss = False

    if 'target_enc_layer_norm' in args['meta'].keys():
        target_enc_layer_norm = args['meta']['target_enc_layer_norm']
    else:
        target_enc_layer_norm = False
    if 'mlp_bias' in args['meta'].keys():
        mlp_bias = args['meta']['mlp_bias']
    else:
        mlp_bias = True

    # Optional Laplacian PE hyperparameters. Only consumed when
    # cell_pos_enc == 'laplacian'; harmlessly ignored otherwise.
    laplacian_k = args['meta'].get('laplacian_k', 8)
    laplacian_sigma = float(args['meta'].get('laplacian_sigma', 1.0))

    # Optional RoPE 2D hyperparameters. Only consumed when
    # cell_pos_enc == 'rope'. ``rope_freq_scale`` defaults to math.pi
    # (in init_model) which gives ~one full oscillation per neighborhood
    # if coords are normalized to [-1, 1] -- tune to your tissue scale.
    rope_freq_scale = args['meta'].get('rope_freq_scale', None)
    rope_rotation_augment = bool(
        args['meta'].get('rope_rotation_augment', True))

    # Optional batch-correction kwargs (off by default; existing
    # behavior preserved). Two independent features:
    #   adaln: per-batch AdaLN modulation inside the encoder + predictor
    #   adv_classifier: gradient-reversal batch classifier on the cell
    #                   embedding (DANN-style)
    # Both read the per-cell batch label directly from the
    # ``batch_value`` column that the dataset (CellGraphDataset /
    # CellNeighborhoodDataset) exposes in every item.
    bc_cfg = args.get('batch_correction', {}) or {}
    adaln_kwargs = bc_cfg.get('adaln', None)
    if adaln_kwargs and not adaln_kwargs.get('enabled', False):
        adaln_kwargs = None
    elif adaln_kwargs is not None:
        adaln_kwargs = dict(adaln_kwargs)
        adaln_kwargs.setdefault('n_batches', bc_cfg.get('n_batches'))
        if adaln_kwargs.get('n_batches') is None:
            raise ValueError(
                "batch_correction.adaln.enabled=True requires "
                "batch_correction.n_batches (or adaln.n_batches) to be set.")
        # Validate scope upfront so a typo blows up at config-load time.
        scope = str(adaln_kwargs.get('scope', 'both')).lower()
        if scope not in ('encoder', 'predictor', 'both'):
            raise ValueError(
                "batch_correction.adaln.scope must be one of "
                f"'encoder', 'predictor', 'both' (got {scope!r}).")
        adaln_kwargs['scope'] = scope

    # Read-Depth-Aware (RDA) conditioning (scFoundation-style).
    # Reads top-level ``rda`` config; encoder-only (no predictor
    # wiring). Disabled by default; zero-init makes RDA-at-step-0 a
    # no-op even when enabled.
    rda_kwargs = args.get('rda', None)
    if rda_kwargs and not rda_kwargs.get('enabled', False):
        rda_kwargs = None
    elif rda_kwargs is not None:
        rda_kwargs = dict(rda_kwargs)

    # Protocol-MoE predictor head. Adds a per-protocol (e.g.
    # per-assay) zero-init learnable bias on top of the predictor
    # output. Routed via a configurable values column.
    moe_kwargs = bc_cfg.get('protocol_moe', None)
    if moe_kwargs and not moe_kwargs.get('enabled', False):
        moe_kwargs = None
    elif moe_kwargs is not None:
        moe_kwargs = dict(moe_kwargs)
        if moe_kwargs.get('routing_index') is None:
            raise ValueError(
                "batch_correction.protocol_moe.enabled=True requires "
                "routing_index (values column for the protocol id).")
        if moe_kwargs.get('n_experts') is None:
            raise ValueError(
                "batch_correction.protocol_moe.enabled=True requires "
                "n_experts.")

    # Cycle-consistency: re-encode each minibatch with random batch
    # labels and penalize the MSE between original and swapped
    # outputs. Encourages batch-invariant encoder features.
    cycle_kwargs = bc_cfg.get('cycle_consistency', None)
    if cycle_kwargs and not cycle_kwargs.get('enabled', False):
        cycle_kwargs = None
    elif cycle_kwargs is not None:
        cycle_kwargs = dict(cycle_kwargs)
        cycle_kwargs.setdefault('n_batches', bc_cfg.get('n_batches'))
        if cycle_kwargs.get('n_batches') is None:
            raise ValueError(
                "batch_correction.cycle_consistency.enabled=True requires "
                "batch_correction.n_batches (or cycle_consistency.n_batches) "
                "to be set.")
    lambda_cycle = float(
        (cycle_kwargs or {}).get('lambda_cycle', 0.0)
    ) if cycle_kwargs else 0.0
    cycle_swap_fraction = float(
        (cycle_kwargs or {}).get('swap_fraction', 1.0)
    ) if cycle_kwargs else 0.0
    cycle_n_batches = int(
        (cycle_kwargs or {}).get('n_batches', 0)
    ) if cycle_kwargs else 0

    adv_classifier_kwargs = bc_cfg.get('adv_classifier', None)
    if adv_classifier_kwargs and not adv_classifier_kwargs.get(
            'enabled', False):
        adv_classifier_kwargs = None
    elif adv_classifier_kwargs is not None:
        adv_classifier_kwargs = dict(adv_classifier_kwargs)
        adv_classifier_kwargs.setdefault(
            'n_batches', bc_cfg.get('n_batches'))
        if adv_classifier_kwargs.get('n_batches') is None:
            raise ValueError(
                "batch_correction.adv_classifier.enabled=True requires "
                "batch_correction.n_batches (or adv_classifier.n_batches) "
                "to be set.")

    lambda_adv = float(
        (adv_classifier_kwargs or {}).get('lambda_adv', 0.1))
    grl_alpha = float(
        (adv_classifier_kwargs or {}).get('grl_alpha', 1.0))

    # Optional distribution-alignment (CORAL / MMD) loss between
    # batches in each minibatch. Non-adversarial alternative to the
    # batch classifier -- no arms race, no collapse attractor.
    dist_align_kwargs = parse_distribution_alignment_kwargs(args)
    lambda_align = float(
        (dist_align_kwargs or {}).get('lambda', 0.0)
    ) if dist_align_kwargs else 0.0

    # Optional protein-embedding initialization for the gene-token
    # embedding layer (UCE-style: frozen ESM matrix + learnable
    # projection). Enabled by a top-level `protein_init` block in the
    # config; absent or `enabled: false` falls back to the default
    # learnable nn.Embedding.
    protein_init_kwargs = parse_protein_init_kwargs(args)

    n_contexts = args['mask']['n_contexts']
    n_targets = args['mask']['n_targets']
    block_masking = args['mask']['block_masking']
    cell_masking = args['mask']['cell_masking']
    context_mask_size = args['mask']['context_mask_size']
    target_mask_size = args['mask']['target_mask_size']
    per_block_mask_ratio = args['mask']['per_block_mask_ratio']
    if 'restrict_special_attention' in args['mask'].keys():
        restrict_special_attention = args['mask']['restrict_special_attention']
    else:
        restrict_special_attention = False
    if 'special_token_pad_ratio' in args['mask'].keys():
        special_token_pad_ratio = args['mask']['special_token_pad_ratio']
    else:
        special_token_pad_ratio = 0.0
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
    # VICReg granularity:
    #   'token' (default, I-JEPA convention): each gene-token embedding
    #           counts as one sample. N = total gene tokens in minibatch.
    #   'cell'  (original VICReg convention): mean-pool gene tokens per
    #           cell first, then apply var/cov on the (B, D) cell-level
    #           embedding. One sample per cell. Aligned with the
    #           adversarial classifier / distribution alignment recipe.
    vicreg_granularity = str(args['optimization'].get(
        'vicreg_granularity', 'token')).lower()
    if vicreg_granularity not in ('token', 'cell'):
        raise ValueError(
            "optimization.vicreg_granularity must be 'token' or 'cell' "
            f"(got {vicreg_granularity!r}).")
    if lambda_var > 0 or lambda_cov > 0:
        logger.info(
            f"[VICReg] lambda_var={lambda_var}, lambda_cov={lambda_cov}, "
            f"granularity='{vicreg_granularity}'. Computed on encoder "
            "output, gene tokens only, float32-cast for numerical "
            "stability under bfloat16 autocast.")
        if vicreg_granularity == 'cell':
            logger.info(
                "[VICReg] cell mode: per-cell embeddings are "
                "all-gathered across DDP ranks before var/cov so "
                f"the effective batch is world_size ({world_size}) x "
                "local_batch. Watch vicreg/n_samples in WandB; if "
                "it is < ~2*enc_emb_dim, the cov estimate is still "
                "rank-deficient and cov_loss may be noisy -- either "
                "raise batch size, use more GPUs, or set lambda_cov=0.")

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
    # Inject the loaded token_dict into protein_init_kwargs so the
    # encoder can align ESM rows to NicheJEPA token IDs.
    if protein_init_kwargs is not None:
        protein_init_kwargs['token_dict'] = token_dict
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

    # VICReg cell-mode sample-size sanity check.
    # In 'cell' granularity, the covariance loss requires the
    # effective sample count (world_size * batch_size, post-gather)
    # to exceed the feature dimension; otherwise the empirical cov
    # is rank-deficient and the off-diagonal gradient is noise.
    # The previous failure mode here was silent: training would
    # appear to run but the encoder learned nothing useful. Hard-
    # error so misconfigured runs fail loud. To override (e.g. you
    # know you need lambda_cov for variance regularization context
    # even at small N), set ``vicreg_allow_underdetermined_cov:
    # True`` in the optimization block.
    if vicreg_granularity == 'cell' and lambda_cov > 0:
        effective_n = world_size * batch_size
        allow_underdetermined = bool(args['optimization'].get(
            'vicreg_allow_underdetermined_cov', False))
        if effective_n <= enc_emb_dim and not allow_underdetermined:
            raise ValueError(
                "VICReg cell-mode covariance loss is statistically "
                "ill-posed at the configured scale: "
                f"effective_N = world_size ({world_size}) * "
                f"batch_size ({batch_size}) = {effective_n}, but "
                f"enc_emb_dim = {enc_emb_dim}. The empirical "
                f"covariance matrix (D x D = {enc_emb_dim}x{enc_emb_dim}) "
                "has rank at most effective_N - 1, so its "
                "off-diagonal entries are dominated by sampling "
                "noise. Fix one of:\n"
                f"  (1) Increase world_size or batch_size so that "
                f"world_size * batch_size > {enc_emb_dim} (ideally "
                f">= {2 * enc_emb_dim}).\n"
                "  (2) Set optimization.vicreg_granularity: 'token' "
                "(thousands of samples, well-conditioned).\n"
                "  (3) Set optimization.lambda_cov: 0 (keep the "
                "variance hinge, drop the decorrelation term).\n"
                "  (4) Set "
                "optimization.vicreg_allow_underdetermined_cov: "
                "True to suppress this check (you accept noisy "
                "cov gradients)."
            )
        if effective_n <= enc_emb_dim and allow_underdetermined:
            logger.warning(
                f"[VICReg] cell mode with effective_N={effective_n} "
                f"<= enc_emb_dim={enc_emb_dim}: cov estimate is "
                "rank-deficient. Proceeding because "
                "vicreg_allow_underdetermined_cov=True. The cov "
                "gradient may be noisy.")

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
        nz_spc=nz_spc,
        new_spc=new_spc,
        mlp_bias=mlp_bias,
        protein_init_kwargs=protein_init_kwargs,
        laplacian_k=laplacian_k,
        laplacian_sigma=laplacian_sigma,
        rope_freq_scale=rope_freq_scale,
        rope_rotation_augment=rope_rotation_augment,
        adaln_kwargs=adaln_kwargs,
        rda_kwargs=rda_kwargs,
        protocol_moe_kwargs=moe_kwargs)
    # Build the adversarial batch classifier head (returns None when
    # disabled, so default training is unchanged).
    batch_classifier = build_batch_classifier_head(
        adv_classifier_kwargs, embed_dim=enc_emb_dim, device=device)
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
            sample_gene_masks=True,
            restrict_special_attention=restrict_special_attention,
            special_token_pad_ratio=special_token_pad_ratio)
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
                persistent_workers=False,
                mega_batch_mult_max=mega_batch_mult_max)
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
            persistent_workers=False,
            mega_batch_mult_max=mega_batch_mult_max)

    print(f"Length of train loader: {len(train_loader)}.")
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

    # Add adversarial batch classifier head params to the optimizer.
    # Use a separate param group so its weight decay can be controlled
    # independently if needed; for now apply the same WD as the
    # encoder's weight-decay group, and no-WD for biases / 1-D params.
    if batch_classifier is not None:
        cls_decay_params, cls_no_decay_params = [], []
        for n, p in batch_classifier.named_parameters():
            if 'bias' in n or p.dim() == 1:
                cls_no_decay_params.append(p)
            else:
                cls_decay_params.append(p)
        if cls_decay_params:
            optimizer.add_param_group({'params': cls_decay_params})
        if cls_no_decay_params:
            optimizer.add_param_group({
                'params': cls_no_decay_params,
                'WD_exclude': True,
                'weight_decay': 0,
            })
        logger.info(
            f"Added {len(cls_decay_params)} decay + "
            f"{len(cls_no_decay_params)} no-decay param groups for the "
            "adversarial batch classifier head.")
    
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
    # Wrap the adversarial batch classifier in DDP too so its grads
    # are reduced across ranks. None when adv classifier is disabled.
    if batch_classifier is not None:
        batch_classifier = DistributedDataParallel(
            batch_classifier,
            static_graph=True,
            device_ids=[LOCAL_RANK],
            output_device=LOCAL_RANK,
            gradient_as_bucket_view=True,
            broadcast_buffers=False)

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

    # Flag for the one-time Laplacian PE diagnostic. Using a flag
    # rather than `epoch == 0 and itr == 0` so it still fires when
    # training is resumed from a checkpoint (where start_epoch != 0).
    _laplacian_diag_logged = False

    # One-shot runtime sanity check for VICReg cell mode. Same
    # rationale as _laplacian_diag_logged: survives checkpoint resume.
    _vicreg_runtime_checked = False

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

        for itr, (udata, masks_enc, masks_pred, masks_attention, pad_special_tokens) in enumerate(train_loader):
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

            # Log Laplacian-PE graph diagnostics once, on the first
            # batch the process actually sees. Survives checkpoint
            # resumes (no epoch==0 / itr==0 condition). Static
            # diagnostic; depends only on the data distribution, not
            # on training state. Useful for sigma / k tuning.
            if (WORLD_RANK == 0
                    and not _laplacian_diag_logged
                    and cell_pos_enc == 'laplacian'):
                try:
                    # Unwrap DDP + EncoderMultiMaskWrapper.
                    _enc = encoder.module
                    if hasattr(_enc, 'backbone'):
                        _enc = _enc.backbone
                    diag = _enc.compute_laplacian_diagnostic(udata)
                except Exception as exc:
                    logger.exception(
                        "Laplacian PE diagnostic computation failed: %s",
                        exc)
                    diag = {}
                if diag:
                    banner = (
                        "\n"
                        "================ Laplacian PE diagnostic ================\n"
                        + "\n".join(
                            f"  {k:<40s} = {v}"
                            for k, v in diag.items())
                        + "\n"
                        "=========================================================\n"
                        "  Tuning targets:\n"
                        "    laplacian/adj_offdiag_mean in [0.2, 0.7]\n"
                        "    -> if much lower, increase laplacian_sigma\n"
                        "    -> if much higher, decrease laplacian_sigma\n"
                        "    A good first try: laplacian_sigma ~= dist_offdiag_median\n"
                        "========================================================="
                    )
                    logger.info(banner)
                    # Use wandb.log without an explicit step so it
                    # piggybacks on the auto-incremented step counter
                    # used by the regular loss log. An explicit
                    # step=0 can be silently dropped if wandb has
                    # already advanced.
                    try:
                        wandb.log(diag)
                    except Exception as exc:
                        logger.warning(
                            "wandb.log of Laplacian diagnostic failed: %s",
                            exc)
                _laplacian_diag_logged = True

            _new_lr = scheduler.step()
            _new_wd = wd_scheduler.step()

            # Step 1: forward pass
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                # Forward pass of target encoder
                with torch.no_grad():
                    h, _ = target_encoder(
                        batch=udata, masks_attention=masks_attention)
                    if target_enc_layer_norm:
                        h = F.layer_norm(h, (h.size(-1),))
                    h = apply_masks(h, masks_pred, concat=False)

                # Forward pass of context encoder
                z, token_emb = encoder(
                    batch=udata,
                    masks=masks_enc,
                    masks_attention=None)
                # Keep a reference to the encoder's outputs (a list of
                # one tensor per context mask) for the adversarial
                # classifier head. The predictor call below REBINDS
                # ``z`` to its own output, so without this save the
                # encoder output would be inaccessible by the time the
                # adversarial loss is computed.
                z_encoder_output = z

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
                        if excl_spc_from_loss and not pad_special_tokens:
                            loss += F.smooth_l1_loss(zi[:, n_special_tokens:, :], hi[:, n_special_tokens:, :])
                        else:
                            loss += F.smooth_l1_loss(zi, hi)
                    elif loss_fn_type == 'l1':
                        if excl_spc_from_loss and not pad_special_tokens:
                            loss += torch.mean(
                                torch.abs(zi[:, n_special_tokens:, :] - hi[:, n_special_tokens:, :])**loss_exp) / loss_exp
                        else:
                            loss += torch.mean(
                                torch.abs(zi - hi)**loss_exp) / loss_exp
                loss /= len(masks_pred)

                # ----------------------------------------------------
                # VICReg-style variance + covariance regularization
                # ----------------------------------------------------
                # Operates on the ENCODER output ``z_encoder_output``
                # (saved above before the predictor overwrites z),
                # restricted to gene-token positions (special tokens
                # and pad excluded via the JEPA mask indices).
                #
                # Granularity:
                #   'token' -- each gene-token embedding is one sample
                #              (I-JEPA convention; high N, dense signal,
                #              but tokens within a cell are correlated).
                #   'cell'  -- mean-pool gene tokens per cell first;
                #              one sample per cell (original VICReg
                #              convention; matches the granularity used
                #              by the adversarial classifier and the
                #              distribution-alignment losses).
                #
                # Numerical note: under amp.autocast(bfloat16) the
                # encoder output is bfloat16. Computing std and
                # (z.T @ z) on N x D bfloat16 tensors with N in the
                # thousands accumulates large rounding error in the
                # sum dimension, which can corrupt the gradient signal.
                # We explicitly cast to float32 before reducing.
                loss_var = None
                loss_cov = None
                vicreg_n_samples = 0
                if lambda_var > 0 or lambda_cov > 0:

                    # z_encoder_output is a list (one per context mask).
                    enc_z_list = z_encoder_output
                    if not isinstance(enc_z_list, list):
                        enc_z_list = [enc_z_list]

                    # Build the gene-token sample tensor per context
                    # mask. Each mask has its own (B, mask_size) layout,
                    # so we recompute the gene-token filter per mask --
                    # the previous shared-filter version was wrong for
                    # len(masks_enc) > 1.
                    samples_list = []
                    for mi, zi in enumerate(enc_z_list):
                        mask_positions = masks_enc[mi]              # (B, M_i)
                        masked_tokens = torch.gather(
                            udata['tokens'], 1, mask_positions)     # (B, M_i)
                        is_special = mask_positions < n_special_tokens
                        is_pad = (masked_tokens == 0)
                        is_gene = (~is_special) & (~is_pad)         # (B, M_i)

                        if vicreg_granularity == 'token':
                            samples_list.append(zi[is_gene])         # (N_i, D)
                        else:  # 'cell'
                            # Mean-pool gene tokens per cell. Cells
                            # with zero gene tokens after masking are
                            # dropped (no valid mean).
                            gene_count = is_gene.sum(dim=1)          # (B,)
                            zi_masked = (
                                zi * is_gene.unsqueeze(-1).to(zi.dtype))
                            sum_per_cell = zi_masked.sum(dim=1)      # (B, D)
                            denom = gene_count.clamp(min=1).to(
                                zi.dtype).unsqueeze(-1)
                            mean_per_cell = sum_per_cell / denom     # (B, D)
                            valid_local = (gene_count > 0)           # (B,) bool

                            # Cast to float32 BEFORE all-gather so
                            # NCCL ops + downstream var/cov accumulate
                            # in stable precision. (Encoder output is
                            # bfloat16 under autocast.)
                            mean_per_cell_fp32 = mean_per_cell.float()

                            # Critical for DDP: with B = per-GPU
                            # batch size (typically 32-64) and
                            # D = enc_emb_dim (e.g. 768), the empirical
                            # covariance is rank-deficient and the
                            # off-diagonal entries are dominated by
                            # sampling noise. All-gather across ranks
                            # so var/cov see the full global batch
                            # (B * world_size). Same pattern MoCo /
                            # SimCLR / VICReg / Barlow Twins use to
                            # make decorrelation losses work under
                            # DDP. Gradients flow only through this
                            # rank's slot (DDP all-reduce then
                            # averages across ranks).
                            #
                            # NOTE: even with gather, if
                            # B*world_size < ~2*D, the cov estimate
                            # is still rank-deficient and ``lambda_cov``
                            # should be very small (or zero). The
                            # ``vicreg/n_samples`` WandB metric tells
                            # you the effective sample count.
                            mean_gathered = _all_gather_with_local_grad(
                                mean_per_cell_fp32)                   # (B*ws, D)
                            valid_gathered = _all_gather_with_local_grad(
                                valid_local.to(torch.float32)
                            ) > 0.5                                   # (B*ws,)
                            samples_list.append(mean_gathered[valid_gathered])

                    # Concatenate samples across context masks, then
                    # cast to float32 for numerically stable reductions.
                    vicreg_samples = torch.cat(samples_list, dim=0).float()

                    # ----- variance loss (per-feature std hinge) -----
                    def variance_loss(z, gamma=1.0, eps=1e-4):
                        std = z.std(dim=0) + eps
                        return torch.mean(torch.relu(gamma - std))

                    # ----- covariance loss (off-diag squared) -----
                    def covariance_loss(z):
                        z = z - z.mean(dim=0)
                        N, D = z.shape
                        cov = (z.T @ z) / (N - 1)
                        off_diag = cov - torch.diag(torch.diag(cov))
                        return (off_diag ** 2).sum() / D

                    # Guard against degenerate batches (no valid samples,
                    # e.g. all cells had zero gene tokens in the mask).
                    # std/cov need at least 2 samples to be defined.
                    vicreg_n_samples = int(vicreg_samples.size(0))

                    # One-shot runtime sanity check: confirm the
                    # actual post-gather sample count is what the
                    # startup check assumed. If gather silently
                    # failed (e.g. dist not initialized) or a lot of
                    # cells got filtered, this catches it on the
                    # first batch. Errors out unless explicitly
                    # allowed.
                    if (vicreg_granularity == 'cell'
                            and lambda_cov > 0
                            and not _vicreg_runtime_checked):
                        if vicreg_n_samples <= enc_emb_dim and not bool(
                                args['optimization'].get(
                                    'vicreg_allow_underdetermined_cov',
                                    False)):
                            raise RuntimeError(
                                "VICReg cell-mode runtime check: "
                                f"vicreg_samples.size(0) = "
                                f"{vicreg_n_samples} <= enc_emb_dim "
                                f"= {enc_emb_dim} on first VICReg "
                                "iteration. Cov is rank-deficient. "
                                "Either DDP all-gather isn't firing "
                                "(check dist.is_initialized() and "
                                f"world_size = {world_size}), too "
                                "many cells were filtered as "
                                "invalid (empty gene_count after "
                                "masking), or your effective batch "
                                "is genuinely too small. See "
                                "startup banner for fixes.")
                        logger.info(
                            f"[VICReg] cell-mode runtime check passed: "
                            f"vicreg_n_samples={vicreg_n_samples}, "
                            f"enc_emb_dim={enc_emb_dim}, "
                            f"ratio={vicreg_n_samples / enc_emb_dim:.2f}.")
                        _vicreg_runtime_checked = True

                    if vicreg_n_samples >= 2:
                        loss_var = variance_loss(vicreg_samples)
                        loss_cov = covariance_loss(vicreg_samples)
                        loss = (loss
                                + lambda_var * loss_var
                                + lambda_cov * loss_cov)

                # ----------------------------------------------------
                # Adversarial batch classifier (DANN-style)
                # ----------------------------------------------------
                # Pull a single-scalar batch label per cell from the
                # special-token-position value, mean-pool the encoder's
                # context-mask output to a per-cell embedding, send it
                # through the gradient-reversal layer, then a small MLP
                # classifier head. CE loss is ADDED to the total with
                # weight lambda_adv. Gradient reversal pushes the
                # encoder to produce batch-invariant representations.
                adv_loss = None
                adv_accuracy = None
                dist_align_loss_val = None
                dist_align_info = None
                # `cell_emb` is shared between the adversarial path
                # and the distribution-alignment path. Computed lazily:
                # set to None here and populated in whichever block
                # fires first.
                cell_emb = None
                if batch_classifier is not None:
                    cell_emb = mean_pool_cell_embedding(
                        z_encoder_output[0],
                        n_special_tokens=n_special_tokens,
                    )
                    cell_emb_rev = grad_reverse(cell_emb, alpha=grl_alpha)
                    batch_logits = batch_classifier(cell_emb_rev)
                    # Per-cell batch label from the values tensor.
                    # By convention the batch token is the first
                    # special token in the sequence, so its value
                    # (an 0-indexed batch class id) lives at index 0.
                    batch_label = udata['values'][:, 0].long()
                    n_cls = batch_logits.size(-1)
                    # Hard range check. Silent clamping collapsed all
                    # out-of-range cells into the last class and let
                    # bugs go undetected. Raise loudly so the user
                    # can fix n_batches in their config. Note that
                    # values[:, 0] is offset-subtracted across all
                    # spv_* tokens (NOT 0-indexed across batches), so
                    # for a small dataset n_batches typically needs
                    # to be ~n_special_values + 2.
                    with torch.no_grad():
                        max_obs = int(batch_label.max().item())
                        min_obs = int(batch_label.min().item())
                    if max_obs >= n_cls or min_obs < 0:
                        raise RuntimeError(
                            "Adversarial batch classifier: batch labels "
                            f"in range [{min_obs}, {max_obs}] but "
                            f"n_batches (classifier output dim) = {n_cls}. "
                            "Set batch_correction.n_batches to at least "
                            f"{max_obs + 1}."
                        )
                    adv_loss = F.cross_entropy(batch_logits, batch_label)
                    loss = loss + lambda_adv * adv_loss
                    with torch.no_grad():
                        adv_accuracy = (
                            batch_logits.argmax(dim=-1) == batch_label
                        ).float().mean().item()

                # ----------------------------------------------------
                # Distribution alignment (CORAL / MMD)
                # ----------------------------------------------------
                # Non-adversarial batch correction. Pulls per-batch
                # embedding distributions toward each other without a
                # classifier. Safe to combine with VICReg variance
                # loss (recommended) so the alignment can't be
                # trivially satisfied by collapsing all batches to a
                # single point.
                dist_align_loss_val = None
                dist_align_info = None
                if dist_align_kwargs is not None and lambda_align > 0:
                    # Per-cell embedding (same recipe as for the
                    # adversarial classifier). Pool over non-special
                    # token positions.
                    if 'cell_emb' not in locals() or cell_emb is None:
                        cell_emb_for_align = mean_pool_cell_embedding(
                            z_encoder_output[0],
                            n_special_tokens=n_special_tokens,
                        )
                    else:
                        cell_emb_for_align = cell_emb
                    batch_label_align = udata['values'][:, 0].long()
                    dist_align_loss_val, dist_align_info = (
                        compute_distribution_alignment_loss(
                            cell_emb=cell_emb_for_align,
                            batch_label=batch_label_align,
                            method=dist_align_kwargs['method'],
                            mmd_sigmas=dist_align_kwargs['mmd_sigmas'],
                            sinkhorn_eps=dist_align_kwargs['sinkhorn_eps'],
                            sinkhorn_n_iter=(
                                dist_align_kwargs['sinkhorn_n_iter']),
                            max_pairs=dist_align_kwargs['max_pairs'],
                        )
                    )
                    loss = loss + lambda_align * dist_align_loss_val

                # ----------------------------------------------------
                # Cycle consistency (batch-swap)
                # ----------------------------------------------------
                # With probability ``swap_fraction``, re-encode the
                # current minibatch with random batch labels and
                # penalize the MSE between original and swapped
                # encoder outputs. Stochastic gating keeps the
                # double-forward cost manageable.
                cycle_loss_val = None
                cycle_n_changed = 0
                if (cycle_kwargs is not None
                        and lambda_cycle > 0
                        and cycle_swap_fraction > 0
                        and torch.rand(()).item() < cycle_swap_fraction):
                    swapped_batch, changed_mask = make_swapped_batch(
                        udata, n_batches=cycle_n_batches)
                    cycle_n_changed = int(changed_mask.sum().item())
                    if cycle_n_changed > 0:
                        z_swapped, _ = encoder(
                            batch=swapped_batch,
                            masks=masks_enc,
                            masks_attention=None)
                        cycle_loss_val = cycle_consistency_loss(
                            z_encoder_output, z_swapped,
                            changed_mask=changed_mask)
                        loss = loss + lambda_cycle * cycle_loss_val

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
                log_payload = {
                    "loss": float(loss),
                    "lr": float(_new_lr),
                    "epoch": int(epoch),
                    "global_norm_enc": float(grad_stats.global_norm),
                    "global_norm_pred": float(grad_stats_pred.global_norm),
                }
                # Diagnostic metrics for the adversarial batch
                # classifier when it's enabled. Target accuracy is
                # ~1/n_batches (random chance): the closer it gets,
                # the more the encoder has succeeded at producing
                # batch-invariant representations.
                if adv_loss is not None:
                    log_payload["adv/batch_classifier_loss"] = float(
                        adv_loss.detach())
                    log_payload["adv/batch_classifier_accuracy"] = float(
                        adv_accuracy)
                # Distribution-alignment diagnostics. Target: loss
                # decreasing over training. n_batches_in_minibatch
                # shows how many distinct batches the alignment loss
                # actually had to work with -- if it's small, the
                # signal is weak (consider increasing batch_size).
                if dist_align_loss_val is not None:
                    log_payload[
                        f"align/{dist_align_kwargs['method']}_loss"
                    ] = float(dist_align_loss_val.detach())
                    if dist_align_info is not None:
                        log_payload["align/n_batches_in_minibatch"] = int(
                            dist_align_info["n_batches_in_minibatch"])
                        log_payload["align/n_pairs"] = int(
                            dist_align_info["n_pairs"])
                if cycle_loss_val is not None:
                    log_payload["cycle/loss"] = float(
                        cycle_loss_val.detach())
                    log_payload["cycle/n_changed"] = int(cycle_n_changed)
                if loss_var is not None:
                    log_payload["vicreg/var_loss"] = float(loss_var.detach())
                    log_payload["vicreg/n_samples"] = int(vicreg_n_samples)
                    log_payload["vicreg/granularity"] = vicreg_granularity
                if loss_cov is not None:
                    log_payload["vicreg/cov_loss"] = float(loss_cov.detach())
                wandb.log(log_payload)
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