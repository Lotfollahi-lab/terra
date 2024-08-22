import sys
import yaml
import pandas as pd
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel
import os
import copy
import logging

from .masks.multigene import MaskCollator
from .utils.distributed import init_distributed
from .utils.logging import CSVLogger
from .datasets.cell_neighborhood_dataset import make_cell_neighborhood_dataset
from .helper import load_checkpoint, init_model
from tqdm import tqdm
import anndata
from .utils.eval_utils  import process_loader
from .utils.emb_utils import calculate_sequence_length, create_and_save_anndata
from .utils.config_utils import generate_output_name

# Set global seed
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def evaluation(args, train_dataset, test_dataset, resume_preempt=False):
    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    r_file = args['meta']['read_checkpoint']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    enc_depth = args['meta']['enc_depth']
    enc_emb_dim = args['meta']['enc_emb_dim']

    # Set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    batch_size = args['data']['batch_size']
    vocab_size = args['data']['vocab_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    data_set_name = args['data']['data_set_name']
    num_epochs = args['optimization']['epochs']
    seq_len_cell = args['data']['seq_len_cell']
    seq_len_neighborhood = args['data']['seq_len_neighborhood']
    just_cell = args['data']['just_cell']
    just_neighborhood = args['data']['just_neighborhood']
    has_cls = args['data']['has_cls']
    learnable = args['optimization']['learnable']
    
    # Compute seq_len based on different configuration files.
    seq_len = calculate_sequence_length(just_cell, just_neighborhood, seq_len_cell, seq_len_neighborhood, has_cls)

    # -- MASK
    n_targets = args['mask']['n_targets']
    n_contexts = args['mask']['n_contexts']
    target_mask_size = args['mask']['target_mask_size']
    context_mask_size = args['mask']['context_mask_size']


    # -- LOGGING

    # Set the folder for saving extracted features
    folder = (f"logs/{data_set_name}_"
               f"pred_depth_{pred_depth}_pred_emb_dim_{pred_emb_dim}_"
               f"enc_depth_{enc_depth}_n_targets_{n_targets}_"
               f"n_contexts_{n_contexts}_target_mask_size_{target_mask_size}_"
               f"context_mask_size_{context_mask_size}_num_epochs_{num_epochs}")
    save_folder = f"{folder}/extracted_features"
    feature_path = f"{save_folder}/" + generate_output_name(args)

    os.makedirs(save_folder, exist_ok=True)
    tag = args['logging']['write_tag']
    dump = os.path.join(folder, f'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)

    # -- log/checkpointing paths
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # Initialize encoder, predictor, and target encoder
    encoder, predictor = init_model(
        device=device,
        seq_len=seq_len,
        enc_emb_dim=enc_emb_dim,
        enc_depth=enc_depth,
        vocab_size =vocab_size,
        pred_depth=pred_depth,
        pos_learnable=learnable,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        has_cls=has_cls)

    target_encoder = copy.deepcopy(encoder)
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)

    # Initialize mask collator
    mask_collator = MaskCollator(seq_len=seq_len,
                                 target_mask_size=target_mask_size,
                                 context_mask_size=context_mask_size,
                                 n_targets=n_targets,
                                 n_contexts=n_contexts)

    # Initialize dataloader and -sampler
    _, train_loader = make_cell_neighborhood_dataset(
        batch_size=batch_size,
        data=train_dataset,
        vocab_size=vocab_size,
        seq_len=seq_len,
        collator=mask_collator,
        pin_mem=pin_mem,
        training=True,
        num_workers=num_workers,
        just_cell=just_cell,
        just_neighborhood=just_neighborhood,
        seq_len_cell = seq_len_cell,
        seq_len_neighborhood = seq_len_neighborhood,
        has_cls = has_cls,
        distributed=False)
    _, test_loader = make_cell_neighborhood_dataset(
        batch_size=batch_size,
        data=test_dataset,
        vocab_size=vocab_size,
        seq_len=seq_len,
        collator=mask_collator,
        pin_mem=pin_mem,
        training=False,
        num_workers=num_workers,
        just_cell=just_cell,
        just_neighborhood=just_neighborhood,
        seq_len_cell = seq_len_cell,
        seq_len_neighborhood = seq_len_neighborhood,
        has_cls = has_cls,
        distributed=False)
    
    _,_, target_encoder,_,_, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=None,
            scaler=None)
    #Extract Features.
    target_encoder.eval()
    all_features = []
    all_obs = []
    process_loader(target_encoder, train_loader, args, 'train', all_features=all_features, all_obs=all_obs)
    process_loader(target_encoder, test_loader, args, 'test', all_features=all_features, all_obs=all_obs)
    #Save and return anndata
    return create_and_save_anndata(all_features, all_obs, output_file=feature_path)

