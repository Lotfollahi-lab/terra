import argparse
import logging
import os
import random
from datetime import datetime

import anndata as ad
import torch

from src.terra.datasets.prepare_dataset import prepare_dataset
from src.terra.infer import infer
from src.terra.utils.config import create_params_from_YAML_wandb_config
from src.terra.utils.distributed import init_distributed


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run TERRA inference.')
    parser.add_argument('--fname', type=str, default='configs.yaml',
                        help='Name of the config file to load.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    logging.basicConfig()
    logger = logging.getLogger()

    rank = 0
    world_size = 1
    world_size, rank = init_distributed(
        rank_and_world_size=(rank, world_size), port=random.randint(40000, 50000))
    logger.info(f'Running... (rank: {rank}/{world_size})')

    params = create_params_from_YAML_wandb_config(
        args.fname,
        logger)

    artifact_folder_path = '../terra-reproducibility/artifacts'

    folder_path = os.path.join(artifact_folder_path,
                               params['data']['data_set_name'],
                               params['meta']['load_timestamp'])

    train_dataset, test_dataset = prepare_dataset(params)
    train_data = infer(params,
                       train_dataset,
                       agg_type=params['embedding']['agg_type'],
                       agg_excluded_tokens=None,
                       cell_gene_ids=[],
                       neighborhood_gene_ids=[],
                       load_folder_path=folder_path)
    test_data = infer(params,
                      test_dataset,
                      agg_type=params['embedding']['agg_type'],
                      agg_excluded_tokens=None,
                      cell_gene_ids=[],
                      neighborhood_gene_ids=[],
                      load_folder_path=folder_path)
    adata_combined = ad.concat(
        [train_data, test_data], axis=0) # concat along the obs (cells)
    adata_combined.write(f'{folder_path}/adata.h5ad')
    print("Finished inference script.")