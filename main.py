"""
Adapted from Assran, M. et al. Self-supervised learning from images with a 
Joint-Embedding Predictive Architecture. Proc. IEEE Comput. Soc. Conf. Comput.
Vis. Pattern Recognit. 15619–15629 (2023); 
https://github.com/facebookresearch/ijepa/blob/main/main.py (05.06.2024).
"""

import argparse
import logging
import multiprocessing as mp
import os
import pprint
import yaml
from datetime import datetime

import anndata as ad
import pandas as pd
import wandb

from src.nichejepa.datasets.utils import prepare_dataset
from src.nichejepa.infer import infer
from src.nichejepa.train import train
from src.nichejepa.utils.config import create_params_from_YAML_wandb_config
from src.nichejepa.utils.distributed import init_distributed
from src.nichejepa.utils.evaluation import clustering_metrics


# Setup argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run NicheJEPA training and evaluation.')
    parser.add_argument('--fname', type=str, default='configs.yaml',
                        help='Name of the config file to load.')
    parser.add_argument('--devices', type=str, nargs='+', default=['cuda:0'],
                        help='Devices to use on the local machine.')
    parser.add_argument('--do_sweep', action='store_true',
                        help='Enable or disable parameter sweeping.')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode.')
    return parser.parse_args()


# Main function to handle training or evaluation per process
def process_main(rank, args, params, world_size, devices, logger, folder_path, is_training=True):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)

    world_size, rank = init_distributed(
        rank_and_world_size=(rank, world_size), port=40003)
    logger.info(f'Running... (rank: {rank}/{world_size})')

    # Execute training or evaluation
    if is_training:
        train_dataset, test_dataset = prepare_dataset(params)
        train(params, train_dataset, test_dataset, save_folder_path=folder_path)
    else:
        train_dataset, test_dataset = prepare_dataset(params)
        train_data = infer(params, train_dataset, load_folder_path=folder_path)
        test_data = infer(params, test_dataset, load_folder_path=folder_path)
        adata_combined = ad.concat(
            [train_data, test_data], axis=0) # concat along the obs (cells)
        adata_combined.write(f'{folder_path}/adata.h5ad')
        cell_type_nmi_ari = clustering_metrics(
            adata_combined,
            emb_key=f"cell_emb_layer_{params['meta']['enc_depth'] - 1}",
            label_col='cell_type')
        '''
        niche_nmi_ari = clustering_metrics(
            adata_combined,
            emb_key=f"neighborhood_emb_layer_{params['meta']['enc_depth'] - 1}",
            label_col='major_brain_region')
        wandb.log(
            {"folder_path": folder_path,
             "niche_nmi": niche_nmi_ari['nmi'],
             "niche_ari": niche_nmi_ari['ari'],
             'cell_type_nmi': cell_type_nmi_ari['nmi'],
             'cell_type_ari': cell_type_nmi_ari['ari']})
        '''

# Function to manage sweeping process
def sweep_func(args):
    num_gpus = len(args.devices)
    processes = []
    
    wandb.init(project='nichejepa-sweep', mode='offline')

    if len(wandb.config.keys()) != 0:
      update_from_sweep = True
    else:      
      update_from_sweep = False

    logging.basicConfig()
    logger = logging.getLogger()

    params = create_params_from_YAML_wandb_config(
        args.fname,
        logger,
        sweep_config=wandb.config,
        update_from_sweep=update_from_sweep)
    logger.info(f'Called with params from {args.fname} and wandb.')

    artifact_folder_path = '../nichejepa-reproducibility/artifacts'
    current_timestamp = (
        datetime.now().strftime("%d%m%Y_%H%M%S") +
        f"_{datetime.now().microsecond // 1000:03d}")
    folder_path = os.path.join(artifact_folder_path,
                            params['data']['dataset_name'],
                            current_timestamp)

    # Run the process_main function in a single or multi-GPU setting
    if args.test:
        process_main(0, args, params, num_gpus, args.devices, logger, folder_path)
    else:
        for rank in range(num_gpus):
            p = mp.Process(target=process_main,
                           args=(rank, args, params, num_gpus, args.devices, logger, folder_path))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()  
    processes = []
    if args.test:
       process_main(0, args, params, 1, [args.devices[0]], logger, folder_path, is_training=False)
    else:
       for rank in range(1):
            p = mp.Process(target=process_main,
                           args=(rank, args, params, 1, [args.devices[0]], logger, folder_path, False))
            p.start()
            processes.append(p)

       for p in processes:
            p.join()

# Entry point of the script
if __name__ == '__main__':
    args = parse_arguments()
    
    # Configuration for W&B sweep
    sweep_config = {
        'method': 'random',
        'metric': {'name': 'niche_nmi', 'goal': 'maximize'},
        'parameters': {
            'enc_pred_depth': {'values': [31,32,41]},
            'pos_learnable': {'values': [1,0]},
            'ema': {'distribution': 'uniform', "max": 1, "min": 0},
            'per_block_mask_ratio': {'distribution': 'uniform',
                                       "max": 0.6, "min": 0.1},
            'n_targets': {'distribution': 'int_uniform', 'min': 1, 'max': 9},
        }
    }

    # Start W&B sweep or single run
    if args.do_sweep:
        sweep_id = wandb.sweep(sweep_config, project='nichejepa-sweep')
        wandb.agent(sweep_id,
                    function=lambda: sweep_func(args=args),
                    count=10000)
    else:
        sweep_func(args=args)
