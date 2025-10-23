import os
import sys
import argparse
import pickle
import logging
import yaml
from typing import Literal

import numpy as np
import anndata as ad
import torch
from tqdm import tqdm

from datasets import Dataset
from datasets import Dataset, load_from_disk

from app.helper import init_model, init_opt, load_checkpoint
from nichejepa.datasets.cell_datasets import init_cell_dataset
from nichejepa.datasets.dataloaders import init_dataloader_and_sampler
from nichejepa.masks.block_masking  import BlockMaskCollator
from nichejepa.masks.cell_masking import CellMaskCollator
from nichejepa.masks.utils import apply_masks
from nichejepa.models.utils import repeat_interleave_batch
from nichejepa.utils.distributed import init_distributed
from nichejepa.utils.logging import (AverageMeter,
                                     CSVLogger,
                                     grad_logger)
from nichejepa.datasets.cell_datasets import CellBaseDataset
from nichejepa.models.modules import ClassificationModel


os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1" # Better error propagation

_GLOBAL_SEED = 0
LOCAL_RANK = None

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


# Setup argument parsing
def parse_arguments():
    """
    Parse config file name from command-line arguments and return hyperparameters in a nested dictionary.
    """
    parser = argparse.ArgumentParser(
        description='Run NicheJEPA finetuning.')
    parser.add_argument('--fname', type=str, default='configs.yaml',
                        help='Name of the config file to load.')
    
    parser_args = parser.parse_args()
    
    # Get the config file name from command line argument
    args_fname = parser_args.fname

    # Read parameters from config file
    with open(args_fname, "r") as f:
        args = yaml.safe_load(f)    

    return args


@torch.no_grad()
def finetune(args: dict,
          dataset: Dataset,
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
    dataset:
        Finetune training dataset.
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
    n_special_values = sum(
        1 for key in token_dict if "spv" in key) # this only works now because of the dummy special values
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

    # Initialize torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}.')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # # Specify last emb layer if not defined
    # if emb_layers is None:
    #     emb_layers = [enc_depth]

    # # Set the folder for saving extracted features
    # save_folder_path = f"{load_folder_path}/extracted_features"
    # feature_path = f"{save_folder_path}/"

    # os.makedirs(save_folder_path, exist_ok=True)

    # # Define checkpointing path
    # latest_path = os.path.join(save_folder_path, f'{write_tag}-latest.pth.tar')
    # load_path = os.path.join(
    #     load_folder_path, r_file) if r_file is not None else latest_path

    # # Initialize target encoder
    # target_encoder, _ = init_model(
    #     gt_type=gt_type,
    #     count_encoding=count_encoding,
    #     n_value_bins=n_value_bins,
    #     cell_pos_enc=cell_pos_enc,
    #     device=device,
    #     vocab_size=vocab_size,
    #     seq_len=seq_len,
    #     n_special_tokens=n_special_tokens,
    #     n_segments=n_segments,
    #     n_special_values=n_special_values,
    #     enc_emb_dim=enc_emb_dim,
    #     enc_depth=enc_depth,
    #     pred_emb_dim=pred_emb_dim,
    #     pred_depth=pred_depth,
    #     num_heads=num_heads,
    #     mlp_ratio=mlp_ratio,
    #     use_flash_attention=use_flash_attention,
    #     use_layer_norm=use_layer_norm,
    #     sep_gene_tokens_neb=sep_gene_tokens_neb)

    # if api_version != 'v3':
    #     return_layer_emb_fn = target_encoder.return_layer_emb
    # else:
    #     return_layer_emb_fn = target_encoder.backbone.return_layer_emb

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

    # Initialize finetune training dataset
    cell_dataset = init_cell_dataset(
            dataset=dataset,
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
            sep_gene_tokens_neb=sep_gene_tokens_neb)

    loader, sampler = init_dataloader_and_sampler(
            cell_dataset=cell_dataset,
            batch_size=batch_size,
            distributed=use_sampler,
            world_size=world_size,
            rank=rank,
            collate_fn=mask_collator,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=True,
            prefetch_factor=4,
            persistent_workers=False)

    # target_encoder = DistributedDataParallel(
    #     target_encoder,
    #     device_ids=[LOCAL_RANK],
    #     output_device=LOCAL_RANK)

    # # Load checkpoint    
    # _, _, target_encoder, _, _, start_epoch, iter_number = load_checkpoint(
    #         device=device,
    #         r_path=load_path,
    #         encoder=None,
    #         predictor=None,
    #         target_encoder=target_encoder,
    #         opt=None,
    #         scaler=None,
    #         is_training=False)
    
    # # Apply PEFT
    # if use_peft:
    #     target_encoder = apply_peft(
    #         target_encoder, peft_method='lora', rank=8)

    # # Convert target encoder to a classification model
    # model = ClassificationModel(
    #     target_encoder, gt_type, num_classes)
    # model.to(device)

    # # Loss function
    # criterion = nn.CrossEntropyLoss()
    
    # # Optimizer (only optimize PEFT parameters)
    # optimizer = torch.optim.Adam(
    #     filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # def save_checkpoint(epoch):
    #         save_dict = {'target_encoder': target_encoder.state_dict(),
    #                      'opt': optimizer.state_dict(),
    #                      'epoch': epoch,
    #                      'zero_epoch_tracking': True,
    #                      'loss': loss_meter.avg,
    #                      'batch_size': batch_size,
    #                      'world_size': world_size,
    #                      'lr': lr}
    #         if rank == 0:
    #             torch.save(save_dict, latest_path)
    #             torch.save(save_dict, save_path.format(epoch=f'ft_{epoch}'))

    # # Run training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch}")
    #     running_loss = 0.0
    #     correct_preds = 0
    #     total_preds = 0

        # Update distributed dataloader epoch
        sampler.set_epoch(epoch)

        for itr, (udata, _, _, masks_attention) in tqdm(enumerate(loader)):
            for key in udata.keys():
                udata[key] = udata[key].to(device, non_blocking=True)
            masks_attention = masks_attention.to(device, non_blocking=True)
            
    #         optimizer.zero_grad()

    #         # Forward pass
    #         logits = model(
    #             udata=udata, masks_attention=masks_attention)

    #         # Compute the loss
    #         loss = criterion(logits, labels)

    #         # Backward pass and optimization
    #         loss.backward()
    #         optimizer.step()

    #         # Track statistics
    #         running_loss += loss.item()
    #         _, predicted = torch.max(logits, 1)
    #         correct_preds += (predicted == labels).sum().item()
    #         total_preds += labels.size(0)

    #     epoch_loss = running_loss / len(loader)
    #     accuracy = correct_preds / total_preds
    #     print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

    # if LOCAL_RANK == 0:
    #     wandb.log(
    #         {"loss": loss,
    #         'lr':_new_lr,
    #         'epoch': epoch,
    #         'global_norm_enc': grad_stats.global_norm,
    #         'global_norm_pred': grad_stats_pred.global_norm,
    #         })
    # assert not np.isnan(loss), 'loss is nan'

    # # Save checkpoint
    # logger.info('avg. loss %.3f' % loss_meter.avg)
    # save_checkpoint(epoch)


if __name__ == '__main__':
    
    # load args dictionary from config file
    args = parse_arguments()
    
    # load finetune training data
    dataset = load_from_disk(args['data']['finetune_training_data_path'])
    cols = [c for c in dataset.column_names if c != 'cell_id']
    dataset.set_format(type="torch", columns=cols, output_all_columns=False)
    
    # finetune model
    finetune(
        args=args,
        dataset=dataset
    )