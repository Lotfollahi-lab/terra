"""
Adapted from Assran, M. et al. Self-supervised learning from images with a Joint-Embedding Predictive Architecture.
Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/main.py (05.06.2024).
"""

import argparse
import multiprocessing as mp
import pprint
import yaml
import os
import logging

import wandb
import pandas as pd

from src.nichejepa.utils.distributed import init_distributed
from src.nichejepa.train import train
from src.nichejepa.infer import infer
from src.nichejepa.utils.config import create_params_from_YAML_wandb_config
from src.nichejepa.datasets.prepare_dataset import prepare_dataset
import anndata as ad
from src.nichejepa.utils.evaluation import clustering_metrics
import multiprocessing as mp

# Setup argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run NicheJEPA training and evaluation.")
    parser.add_argument('--fname', type=str, default='configs.yaml',
                        help='Name of the config file to load')
    parser.add_argument('--devices', type=str, nargs='+', default=['cuda:0'],
                        help='Devices to use on the local machine')
    parser.add_argument('--do_sweep', action='store_true',
                        help='Enable or disable parameter sweeping')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode')
    return parser.parse_args()

# Main function to handle training or evaluation per process
def process_main(rank, args, world_size, devices, is_training=True):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size), port=40004)
    logger.info(f'Running... (rank: {rank}/{world_size})')
    if len(wandb.config.keys())!=0:
      update_from_sweep=True
    else:      
      update_from_sweep=False

    # Execute training or evaluation
    if is_training:
        logger.info(f'Called with params from {args.fname} and wandb')
        params = create_params_from_YAML_wandb_config(args.fname, logger, sweep_config=wandb.config, is_training=is_training, update_from_sweep=update_from_sweep)
        train_dataset, test_dataset = prepare_dataset(params)
        train(params, train_dataset, test_dataset)
    else:
        logger.info(f'Called with params from {args.fname} and wandb')
        params = create_params_from_YAML_wandb_config(args.fname, logger, sweep_config=wandb.config, is_training=is_training, update_from_sweep=update_from_sweep)
        train_dataset, test_dataset = prepare_dataset(params)
        #train_data = infer(params, train_dataset)
        test_data = infer(params, test_dataset)
        print(test_data)
        #adata_combined = ad.concat([train_data, test_data], axis=0)  # Concatenate along the observations (cells)
        if update_from_sweep:
           cell_type_nmi_ari = clustering_metrics(test_data, emb_key='cell_emb_layer_'+str(int(wandb.config.enc_pred_depth // 10)-1),label_col='cell_type')
           niche_nmi_ari = clustering_metrics(test_data, emb_key='neighborhood_emb_layer_'+str(int(wandb.config.enc_pred_depth // 10)-1),label_col='niche')
           wandb.log({"niche_nmi":niche_nmi_ari['nmi'], "niche_ari":niche_nmi_ari['ari'], 'cell_type_ari':cell_type_nmi_ari['ari'], 'cell_type_nmi':cell_type_nmi_ari['nmi']})
        else:
           cell_type_nmi_ari = clustering_metrics(test_data, emb_key='cell_emb_layer_'+str(params['meta']['enc_pred_depth']//10 -1), label_col='cell_type')
           niche_nmi_ari = clustering_metrics(test_data, emb_key='neighborhood_emb_layer_'+str(params['meta']['enc_pred_depth']//10-1), label_col='niche')
           wandb.log({"niche_nmi":niche_nmi_ari['nmi'], "niche_ari":niche_nmi_ari['ari'], 'cell_type_ari':cell_type_nmi_ari['ari'], 'cell_type_nmi':cell_type_nmi_ari['nmi']})
        
# Function to manage sweeping process
def sweep_func(args):
    num_gpus = len(args.devices)
    processes = []
    
    # Initialize W&B for sweeping
    wandb.init(project="nichejepa-sweep")
    # Run the process_main function in a single or multi-GPU setting
    if args.test:
        process_main(0, args, num_gpus, args.devices)
    else:
        for rank in range(num_gpus):
            p = mp.Process(target=process_main, args=(rank, args, num_gpus, args.devices))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()  
    processes = []
    if args.test:
       process_main(0, args, 1, [args.devices[0]], is_training=False)
    else :
       for rank in range(1):
            p = mp.Process(target=process_main, args=(rank, args, 1, [args.devices[0]],False))
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
            'enc_pred_depth': {'values': [31]},
            #'pos_learnable': {'values': [1,0]},
            'ema': {'distribution': 'uniform', "max": 1, "min": 0},
            'per_segment_mask_ratio': {'distribution': 'uniform', "max": 0.6, "min": 0.2},
            'n_targets': {'distribution': 'int_uniform', 'min': 1, 'max': 9}
            #'lr': {'distribution': 'uniform', "max": 0.001, "min": 0.0001}
            
            }
    }

    # Start W&B sweep or single run
    if args.do_sweep:
        sweep_id = wandb.sweep(sweep_config, project="nichejepa-sweep")
        wandb.agent(sweep_id, function=lambda: sweep_func(args=args), count=10000)
    else:
        sweep_func(args=args)

