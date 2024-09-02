import copy
import logging
import os
import sys
import yaml

import anndata
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from .datasets.cell_neighborhood_dataset import make_cell_neighborhood_dataset
from .helper import load_checkpoint, init_model
from .masks.multigene import MaskCollator
from .utils.config_utils import generate_output_name
from .utils.distributed import init_distributed
from .utils.emb_utils import calculate_sequence_length, create_and_save_anndata
from .utils.eval_utils import process_loader
from .utils.logging import CSVLogger


_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def evaluation(args: dict,
               train_dataset,
               test_dataset,
               resume_preempt: bool=False):
    """
    """
    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    r_file = args['meta']['read_checkpoint']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    enc_depth = args['meta']['enc_depth']
    enc_emb_dim = args['meta']['enc_emb_dim']

    # -- DATA
    batch_size = args['data']['batch_size']
    vocab_size = args['data']['vocab_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    data_set_name = args['data']['data_set_name']
    num_epochs = args['optimization']['epochs']
    seq_len_cell = args['data']['seq_len_cell']
    seq_len_neighborhood = args['data']['seq_len_neighborhood']
    incl_cell_seq = args['data']['incl_cell_seq']
    incl_neighborhood_seq = args['data']['incl_neighborhood_seq']
    has_cls = args['data']['has_cls']

    # -- OPTIMIZATION
    learnable = args['optimization']['learnable']

    # -- MASK
    n_targets = args['mask']['n_targets']
    n_contexts = args['mask']['n_contexts']
    target_mask_size = args['mask']['target_mask_size']
    context_mask_size = args['mask']['context_mask_size']

    # Set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    
    # Compute seq_len based on config
    seq_len = calculate_sequence_length(incl_cell_seq,
                                        incl_neighborhood_seq,
                                        seq_len_cell,
                                        seq_len_neighborhood,
                                        has_cls)

    # Set the folder for saving extracted features
    folder = (f"logs/{data_set_name}_"
               f"pred_depth_{pred_depth}_pred_emb_dim_{pred_emb_dim}_"
               f"enc_depth_{enc_depth}_n_targets_{n_targets}_"
               f"n_contexts_{n_contexts}_target_mask_size_{target_mask_size}_"
               f"context_mask_size_{context_mask_size}_num_epochs_{num_epochs}")
    if args['data']['incl_cell_seq']:
       folder += "_incl_cell_seq"
    if args['data']['incl_neighborhood_seq']:
       folder += "_incl_neighborhood_seq"
    specific_cell_types = args['data'].get('specific_cell_types')
    if len(specific_cell_types) != 0:
       subset_name = "_".join(specific_cell_types)
       folder += f"_subset_{subset_name}"
    else:
       folder += "_total"
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
        vocab_size=vocab_size,
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
    mask_collator = MaskCollator(
        n_targets=n_targets,
        n_contexts=n_contexts,
        target_mask_size=target_mask_size,
        context_mask_size=context_mask_size,
        seq_len_cell=seq_len_cell,
        seq_len_neighborhood=seq_len_neighborhood,
        has_cls=has_cls)

    # Initialize dataloader and -sampler
    _, train_loader = make_cell_neighborhood_dataset(
        batch_size=batch_size,
        data=train_dataset,
        vocab_size=vocab_size,
        seq_len=seq_len,
        collator=mask_collator,
        pin_mem=pin_mem,
        num_workers=num_workers,
        incl_cell_seq=incl_cell_seq,
        incl_neighborhood_seq=incl_neighborhood_seq,
        seq_len_cell=seq_len_cell,
        seq_len_neighborhood=seq_len_neighborhood,
        has_cls=has_cls,
        distributed=False)
    _, test_loader = make_cell_neighborhood_dataset(
        batch_size=batch_size,
        data=test_dataset,
        vocab_size=vocab_size,
        seq_len=seq_len,
        collator=mask_collator,
        pin_mem=pin_mem,
        num_workers=num_workers,
        incl_cell_seq=incl_cell_seq,
        incl_neighborhood_seq=incl_neighborhood_seq,
        seq_len_cell=seq_len_cell,
        seq_len_neighborhood=seq_len_neighborhood,
        has_cls=has_cls,
        distributed=False)
    
    _,_, target_encoder,_,_, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=None,
            scaler=None)

    # Extract features
    target_encoder.eval()
    all_features = []
    all_obs = []
    process_loader(target_encoder,
                   train_loader,
                   args,
                   'train',
                   all_features=all_features,
                   all_obs=all_obs)
    process_loader(target_encoder,
                   test_loader,
                   args,
                   'test',
                   all_features=all_features,
                   all_obs=all_obs)

    # Save and return adata with results
    adata = create_and_save_anndata(all_features,
                                    all_obs,
                                    output_file=feature_path)

    return adata
