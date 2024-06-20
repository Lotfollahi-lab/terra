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
from .helper import load_checkpoint, init_model, init_opt
from tqdm import tqdm
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(args, resume_preempt=False):
    # Set the folder for logging
    top_niche = args['mask']['top_niche']
    top_cell_type = args['mask']['top_cell_type']

    for seed in tqdm(range(10)):
        # ----------------------------------------------------------------------- #
        #  PASSED IN PARAMS FROM CONFIG FILE
        # ----------------------------------------------------------------------- #
        # -- log
        folder = args['logging']['folder']+str(seed)+'/'
        save_folder =  args['logging']['save_folder']
        os.makedirs(save_folder, exist_ok=True)
        # -- META
        use_bfloat16 = args['meta']['use_bfloat16']
        model_name = args['meta']['model_name']
        load_model = args['meta']['load_checkpoint'] or resume_preempt
        r_file = args['meta']['read_checkpoint']
        pred_depth = args['meta']['pred_depth']
        pred_emb_dim = args['meta']['pred_emb_dim']
        enc_depth = args['meta']['enc_depth']
        enc_emb_dim = args['meta']['enc_emb_dim']

        if not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)

        # -- DATA
        batch_size = args['data']['batch_size']
        seq_len = args['data']['seq_len']
        vocab_size = args['data']['vocab_size']
        pin_mem = args['data']['pin_mem']
        num_workers = args['data']['num_workers']
        # --

        # -- MASK
        n_targets = args['mask']['n_targets']
        n_contexts = args['mask']['n_contexts']
        target_mask_size = args['mask']['target_mask_size']
        context_mask_size = args['mask']['context_mask_size']

        # -- LOGGING
        tag = args['logging']['write_tag']
        dump = os.path.join(folder, f'params-ijepa.yaml')
        with open(dump, 'w') as f:
            yaml.dump(args, f)
        # ----------------------------------------------------------------------- #

        try:
            mp.set_start_method('spawn')
        except Exception:
            pass

        # Initialize torch distributed backend
        world_size, rank = init_distributed()
        logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
        if rank > 0:
            logger.setLevel(logging.ERROR)

        # -- log/checkpointing paths
        log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
        latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
        load_path = None
        if load_model:
            load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

        # -- make csv_logger
        csv_logger = CSVLogger(log_file,
                               ('%d', 'epoch'),
                               ('%d', 'itr'),
                               ('%.5f', 'loss'),
                               ('%.5f', 'mask-A'),
                               ('%.5f', 'mask-B'),
                               ('%d', 'time (ms)'))

        # Initialize encoder, predictor and target encoder
        encoder, predictor = init_model(
            device=device,
            seq_len=seq_len,
            enc_emb_dim=enc_emb_dim,
            enc_depth=enc_depth,
            vocab_size=vocab_size,
            pred_depth=pred_depth,
            pred_emb_dim=pred_emb_dim,
            model_name=model_name)
        target_encoder = copy.deepcopy(encoder)

        # Initialize mask collator
        mask_collator = MaskCollator(seq_len=seq_len,
                                     target_mask_size=target_mask_size,
                                     context_mask_size=context_mask_size,
                                     n_targets=n_targets,
                                     n_contexts=n_contexts)

        # Initialize dataloader and -sampler
        data_path = args['data']['data_path']
        dataset = load_from_disk(data_path, keep_in_memory=True)
        dataset = dataset.train_test_split(test_size=args['data']['split'], seed=seed)

        _, train_loader, test__sampler = make_cell_neighborhood_dataset(
            batch_size=batch_size,
            data=dataset["train"],
            vocab_size=vocab_size,
            seq_len=seq_len,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            drop_last=True)
        _, test_loader, train__sampler = make_cell_neighborhood_dataset(
            batch_size=batch_size,
            data=dataset["test"],
            vocab_size=vocab_size,
            seq_len=seq_len,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=False,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            drop_last=True)
        encoder = DistributedDataParallel(encoder, static_graph=True)
        start_epoch = 0
        # Load training checkpoint
        ipe = len(train_loader)

        if load_model:
            encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
                device=device,
                r_path=load_path,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
                opt=None,
                scaler=None)
            for _ in range(start_epoch * ipe):
                mask_collator.step()

        encoder.eval()
        data = []

        def process_loader(loader, dataset_type):
            for itr, (udata, masks_enc, masks_pred) in tqdm(enumerate(loader)):
                def load_cell_neighborhoods():
                    # -- unsupervised imgs
                    cell_neighborhood_tokens = udata[0].to(device, non_blocking=True)
                    seg_label = udata[1].to(device, non_blocking=True)
                    niche_label = udata[2]
                    cell_type = udata[3]  # Assuming udata[3] is cell_type
                    masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                    masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                    return (cell_neighborhood_tokens, seg_label, niche_label, cell_type, masks_1, masks_2)

                cell_neighborhood_tokens, seg_label, niche_label, cell_type, masks_enc, masks_pred = load_cell_neighborhoods()

                def eval_step():
                    def forward_context(top_index, label_name, label_value):
                        # Encode all cell neighborhood tokens
                        z = encoder(cell_neighborhood_tokens, seg_label)  # output (B, seq_len, emb_size)
                        masks = (cell_neighborhood_tokens != 0).int()
                        if label_name=='niche_label':
                              masks[:, 0:256] = 0
                        masks[:, top_index:] = 0
                        expanded_mask = masks.unsqueeze(-1).expand_as(z)
                        masked_features = z * expanded_mask
                        summed_features = masked_features.sum(dim=1)
                        count_valid_positions = expanded_mask.sum(dim=1)
                        average_features = summed_features / count_valid_positions.clamp(min=1)
                        average_features[count_valid_positions == 0] = 0
                        average_features = average_features.cpu().numpy()
                        label_cpu = label_value
                        for i in range(len(average_features)):
                            sample_features = average_features[i]
                            sample_label = label_cpu[i]
                            data_dict = {
                                'split': dataset_type,
                                'label_name': label_name,
                                'seed': seed,
                                label_name: sample_label
                            }
                            for j, feature in enumerate(sample_features):
                                data_dict[f'feature_{j}'] = feature
                            data.append(data_dict)

                    with torch.no_grad():
                        forward_context(top_niche, "niche_label", niche_label)
                        forward_context(top_cell_type, "cell_type", cell_type)

                eval_step()

        process_loader(train_loader, 'train')
        process_loader(test_loader, 'test')

        final_df = pd.DataFrame(data)
        final_df.to_csv(os.path.join(save_folder, f'aggregated_features_seed_{seed}.csv'), index=False)

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        args = yaml.safe_load(file)
    main(args)

