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
import pickle
import pprint
import random
import yaml
from datetime import datetime

import anndata as ad
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

from nichejepa.datasets.utils import prepare_dataset
from app.infer import infer
from app.train import train
from nichejepa.utils.config import create_params_from_YAML_wandb_config
from nichejepa.utils.distributed import init_distributed
from nichejepa.utils.evaluation import clustering_metrics


# Setup argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run NicheJEPA training and evaluation.')
    parser.add_argument('--fname', type=str, default='configs.yaml',
                        help='Name of the config file to load.')
    parser.add_argument('--devices', type=str, nargs='+', default=['cuda:0'],
                        help='Devices to use on the local machine.')
    parser.add_argument('--run_id', type=str, default='goodsalad-1',
                        help='Run ID for wandb.')
    parser.add_argument('--do_sweep', action='store_true', default=False,
                        help='Enable or disable parameter sweeping.')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Run in test mode.')
    return parser.parse_args()


# Main function to handle training or evaluation per process
def process_main(rank, args, params, world_size, port, devices, logger, folder_path, is_training=True):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)

    world_size, rank = init_distributed(
        rank_and_world_size=(rank, world_size), port=port)
    logger.info(f'Running... (rank: {rank}/{world_size})')
    logger.info(f'Training mode: {is_training}')

    # Execute training or evaluation
     # Execute training or evaluation
    if is_training:
        train_dataset, val_dataset, test_dataset = prepare_dataset(params)
        train(params, train_dataset, save_folder_path=folder_path)
    else:
        
        train_dataset, _, _ = prepare_dataset(params)
        print(train_dataset)
        with open("top_4250_genes.pkl", "rb") as f:
            cell_gene_ids = pickle.load(f)
            cell_gene_ids = cell_gene_ids[0:500]
            #cell_gene_ids =[]
        with open("top_4250_genes.pkl", "rb") as f:
            neighborhood_gene_ids = pickle.load(f)
            neighborhood_gene_ids = neighborhood_gene_ids[0:500]
        params['state']['read_checkpoint'] = 'nichejepa-ep9.pth.tar'
        #params['state']['read_checkpoint'] = 'nichejepa-ep5_0.pth.tar'
        adata_test = infer(params, train_dataset,load_folder_path='/lustre/scratch126/cellgen/team361/sb75/nichejepa-reproducibility/artifacts/hst_corpus_70m/06052025_160815_325/', agg_type='avg',
                           cell_gene_ids=cell_gene_ids, neighborhood_gene_ids=neighborhood_gene_ids, return_gene=False, return_cosine_sim=True,
                           return_gene_per_data=False, dataset_ids=['1000'], compute_cosine_with='neighborhood', obs_cols=['cell_type','niche_type'])
        adata_test.write("result_06052025_160815_325_cell.h5ad")
        exit()
        
        params['data']['batch_size'] = 128
        params['data']['tokenized_data_folder_path'] = '/lustre/scratch126/cellgen/team361/DATASETS/gold/cell-graph-tokenizer/hst_corpus_70m/hst_corpus_70m_32_None_None_None_None_None_shifted_log_knn_6.dataset'
        params['data']['precomputed_split'] = '/lustre/scratch126/cellgen/team361/DATASETS/tokenizer/hst_corpus_70m_32_None_None_None_None_None_shifted_log_knn_6_precomputed_split_validation.pkl'
        params['data']['precomputed_n_nonzero_tokens'] = '/lustre/scratch126/cellgen/team361/DATASETS/tokenizer/hst_corpus_70m_32_None_None_None_None_None_shifted_log_knn_6_n_nonzero_tokens_validation.pkl'
        params['data']['n_segments'] = 7
        params['data']['seq_len_cell'] = 256
        params['data']['seq_len_neighborhood'] = 1536  
        params['meta']['n_value_bins'] = 100  
        params['meta']['count_encoding'] = 'mlp'  
        params['meta']['enc_depth'] = 12
        params['meta']['num_heads'] = 12
        params['meta']['enc_emb_dim'] = 768
        params['meta']['pred_depth'] = 6
        params['meta']['pred_emb_dim'] = 384
        params['meta']['special_tokens'] = ['assay']
        params['meta']['api_version'] = 'v2'

        train_dataset, _, _ = prepare_dataset(params)
        print(train_dataset)
        with open("top_4250_genes.pkl", "rb") as f:
            cell_gene_ids = pickle.load(f)
            #cell_gene_ids = cell_gene_ids[0:500]
            #cell_gene_ids =[]
        with open("top_4250_genes.pkl", "rb") as f:
            neighborhood_gene_ids = pickle.load(f)
            #neighborhood_gene_ids = neighborhood_gene_ids[0:500]
        params['state']['read_checkpoint'] = 'nichejepa-ep1_10.pth.tar'
        adata_test = infer(params, train_dataset,load_folder_path='/lustre/scratch126/cellgen/team361/sb75/nichejepa-reproducibility/artifacts/hst_corpus_70m/21032025_125422_751', agg_type='avg',
                          cell_gene_ids=cell_gene_ids, neighborhood_gene_ids=neighborhood_gene_ids, return_gene=False, return_cosine_sim=True,
                           dataset_ids=['1000'], compute_cosine_with= 'cell', obs_cols=['cell_type','niche_type'])
        adata_test.write("result_previous_cell.h5ad")
        exit()
    
        test_data = infer(params, test_dataset, load_folder_path=folder_path)
        #adata_combined = ad.concat(
        #    [train_data, test_data], axis=0) # concat along the obs (cells)
        #adata_combined.write(f'{folder_path}/adata.h5ad')
        cell_type_nmi_ari = clustering_metrics(
            test_data,
            emb_key=f"cell_emb_layer_{params['meta']['enc_depth'] - 1}",
            label_col='cell_type')
        print(f"neighborhood_emb_layer_{params['meta']['enc_depth'] - 1}")
        niche_nmi_ari = clustering_metrics(
            test_data,
            emb_key=f"neighborhood_emb_layer_{params['meta']['enc_depth'] - 1}",
            label_col='niche_type')
        wandb.log(
            {"folder_path": folder_path,
             "niche_nmi": niche_nmi_ari['nmi'],
             "niche_ari": niche_nmi_ari['ari'],
             'cell_type_nmi': cell_type_nmi_ari['nmi'],
             'cell_type_ari': cell_type_nmi_ari['ari'],
             'loss_val' : loss_val,
             'len_dataset': len(test_data)})

# Function to manage sweeping process
def sweep_func(args):
    num_gpus = len(args.devices)
    processes = []
    
    wandb.init(project='nichejepa-sweep', id=args.run_id, resume="allow", group="multi_node_training", mode='online')

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
    logger.info(f'Timestamp {current_timestamp}.')
    print('Timestamp:', current_timestamp)
    print('Params:', params)
    if params['state']['folder_path'] is None:
        folder_path = os.path.join(artifact_folder_path,
                        params['data']['dataset_name'],
                        current_timestamp)
    else:
        folder_path = params['state']['folder_path']

    port = random.randint(40000, 50000)

    # Run the process_main function in a single or multi-GPU setting
    if args.test:
        process_main(0, args, params, num_gpus, port, args.devices, logger, folder_path)
    else:

        for rank in range(num_gpus):
            p = mp.Process(target=process_main,
                           args=(rank, args, params, num_gpus, port, args.devices, logger, folder_path))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()  
    processes = []
    if args.test:
       process_main(0, args, params, 1, port, [args.devices[0]], logger, folder_path, is_training=False)
    else:
       for rank in range(1):
            p = mp.Process(target=process_main,
                           args=(rank, args, params, 1, port, [args.devices[0]], logger, folder_path))
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
        'metric': {'name': 'loss_val', 'goal': 'minimize'},
        'parameters': {
            #'enc_pred_depth': {'values': [31]},
            #'pos_learnable': {'values': [1,0]},
            'ema': {'distribution': 'uniform', "max": 0.75, "min": 0.4},
            'per_block_mask_ratio': {'distribution': 'uniform',
                                       "max": 0.7, "min": 0.4},
            #'n_targets': {'distribution': 'int_uniform', 'min': 1, 'max': 9},
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
