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
from src.nichejepa.train_sweep import train_main
from src.nichejepa.eval_sweep import eval_main
from src.nichejepa.utils.config_utils import create_params_from_YAML_wandb_config

# Setup argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run NicheJEPA training and evaluation.")
    parser.add_argument('--fname', type=str, default='configs.yaml',
                        help='Name of the config file to load')
    parser.add_argument('--devices', type=str, nargs='+', default=['cuda:0'],
                        help='Devices to use on the local machine')
    parser.add_argument('--seed', type=int,
                        help='Seed value for random initialization')
    parser.add_argument('--do_sweep', action='store_true',
                        help='Enable or disable parameter sweeping')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode')
    parser.add_argument('--task', type=str, required=True,
                        help='Name of the task to perform')

    return parser.parse_args()

# Main function to handle training or evaluation per process
def process_main(rank, args, world_size, devices, data, train=True):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    logger.info(f'Called with params from {args.fname}')
    
    create_params_from_YAML_wandb_config(wandb.config, args, logger)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size), port=40316)
    logger.info(f'Running... (rank: {rank}/{world_size})')

    # Execute training or evaluation
    if train:
        train_main(args=params, rank=rank)
    else:
        eval_main(params)

# Function to manage sweeping process
def sweep_func(args):
    num_gpus = len(args.devices)
    manager = mp.Manager()
    data = manager.list()
    processes = []

    # Initialize W&B for sweeping
    if not args.do_sweep:
        config = {
            'pred_enc_depth': 43,
            "learnable": 1,
            "ema": 0.999,
            "context_mask_size": 1100,
            'n_targets': 4,
            'epochs': 2,
            'top_k': 127,
            'top_layer': 4,
            'enc_emb_dim': 768,
        }
        wandb.init(project="nichejepa-sweep", config=config)
    else:
        wandb.init(project="nichejepa-sweep")

    # Run the process_main function in a single or multi-GPU setting
    if args.test:
        data = []
        process_main(0, args, num_gpus, args.devices, data)
    else:
        for rank in range(num_gpus):
            p = mp.Process(target=process_main, args=(rank, args, num_gpus, args.devices, data))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    # Final evaluation after sweeping
    process_main(0, args, 1, [args.devices[0]], data, train=False)

# Entry point of the script
if __name__ == '__main__':
    args = parse_arguments()

    # Configuration for W&B sweep
    sweep_config = {
        'method': 'random',
        'metric': {'name': 'nmi_score', 'goal': 'maximize'},
        'parameters': {
            'pred_enc_depth': {'values': [31]},
            'learnable': {'values': [1]},
            'ema': {'distribution': 'uniform', "max": 1, "min": 0},
            'enc_emb_dim': {'values': [768]},
            'context_mask_size': {'distribution': 'int_uniform', 'min': 300, 'max': 786},
            'n_targets': {'distribution': 'int_uniform', 'min': 1, 'max': 9},
            'target_mask_size': {'distribution': 'int_uniform', 'min': 10, 'max': 30},
            'epochs': {'distribution': 'int_uniform', 'min': 20, 'max': 40},
            'top_layer': {'distribution': 'int_uniform', 'min': 1, 'max': 3}
        }
    }

    # Start W&B sweep or single run
    if args.do_sweep:
        sweep_id = wandb.sweep(sweep_config, project="nichejepa-sweep")
        wandb.agent(sweep_id, function=lambda: sweep_func(args=args), count=10000)
    else:
        sweep_func(args=args)

