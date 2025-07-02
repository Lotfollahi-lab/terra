import os
import torch

import logging
import torch.distributed as dist
from app.train import train
from nichejepa.datasets.utils import prepare_dataset
from nichejepa.utils.config import create_params_from_YAML_wandb_config
import wandb
import sys
# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import argparse
import time
import importlib
import warnings
from datetime import datetime
from datetime import timedelta
import random
from socket import gethostname

warnings.filterwarnings("ignore")
# logger
logging.basicConfig()
logger = logging.getLogger()

# ==========================

# Function to retrieve and log distributed environment variables
def get_distributed_info():
    """
    Retrieves distributed training environment variables and logs them.

    Returns:
        tuple: (WORLD_RANK, LOCAL_RANK, WORLD_SIZE)
    """
    if "SLURM_PROCID" in os.environ:
        WORLD_RANK = int(os.environ["SLURM_PROCID"])
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        GPUS_PER_NODE = int(os.environ["SLURM_GPUS_ON_NODE"])
        assert GPUS_PER_NODE == torch.cuda.device_count()
        print(f"Hello from rank {WORLD_RANK} of {WORLD_SIZE} on {gethostname()} where there are" \
            f" {GPUS_PER_NODE} allocated GPUs per node.", flush=True)
        LOCAL_RANK = WORLD_RANK - GPUS_PER_NODE * (WORLD_RANK // GPUS_PER_NODE)
    if "LOCAL_RANK" in os.environ:
        # Environment variables set by torch.distributed.launch or
        # torchrun
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        WORLD_RANK = int(os.environ["RANK"])
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        # Environment variables set by mpirun
        LOCAL_RANK = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        WORLD_SIZE = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        WORLD_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])
    else:
        import sys
        sys.exit("Can't find the environment variables for local rank")

    # Print the ranks
    print(f"World rank: {WORLD_RANK}, Local rank: {LOCAL_RANK}, World size: {WORLD_SIZE}")

    return WORLD_RANK, LOCAL_RANK, WORLD_SIZE


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def main():
    # Retrieve distributed environment variables
    WORLD_RANK, LOCAL_RANK, WORLD_SIZE = get_distributed_info()

    # Argument parsing (as in your original script)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--backend",
        type=str,
        help="Backend for distributed training.",
        default="nccl",
        choices=["nccl", "gloo", "mpi"],
    )
    parser.add_argument(
        '--fname', 
        type=str, 
        default='configs.yaml',
        help='Name of the config file to load.',
    )

    args = parser.parse_args()
    backend = args.backend
    experiment_name = os.environ.get('EXPERIMENT_NAME')
    run_name = os.environ.get('RUN_NAME')
    run_id = f"{experiment_name}_{run_name}"

    params = create_params_from_YAML_wandb_config(
        args.fname,
        logger)
    logger.info(f'Called with params from {args.fname}.')
    logger.info(f'Params: {params}.')
    if WORLD_RANK==0:
        wandb.init(
            project='nichejepa-pretraining',
            id=run_id,
            resume="allow",
            group="multi_node_training",
            mode='online')
        artifact_folder_path = '../nichejepa-reproducibility/artifacts'
        current_timestamp = (
            datetime.now().strftime("%d%m%Y_%H%M%S") +
            f"_{datetime.now().microsecond // 1000:03d}")
        print(f'Run timestamp: {current_timestamp}.')
        if not wandb.run.resumed:
            wandb.config.run_timestamp = current_timestamp
        print(params)
        if params['state']['folder_path'] is None:
            folder_path = os.path.join(artifact_folder_path,
                                       params['data']['dataset_name'],
                                       current_timestamp)
        else:
            folder_path = params['state']['folder_path']
    else:
        folder_path=None

    print(f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")

    torch.cuda.set_device(LOCAL_RANK)

    # Initialize the distributed backend
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        rank=WORLD_RANK,
        world_size=WORLD_SIZE,
    )

    dist.barrier()
    setup_for_distributed(LOCAL_RANK == 0)

    train_dataset, val_dataset, test_dataset = prepare_dataset(params)
    train(params,
          train_dataset,
          test_dataset,
          save_folder_path=folder_path,
          LOCAL_RANK=LOCAL_RANK,
          WORLD_RANK=WORLD_RANK)


if __name__ == "__main__":
    # Print Torch Version
    print(f"torch.__version__: {torch.__version__}")
    # Print torch CUDA version
    print(f"torch.version.cuda: {torch.version.cuda}")
    # Print torch nccl version
    try:
        nccl_version = torch.cuda.nccl.version()
    except AttributeError:
        nccl_version = "NCCL not available."
    print(f"torch.cuda.nccl.version(): {nccl_version}")
    # Print visible CUDA devices
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count():", torch.cuda.device_count())

    # Set additional environment variables
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    # Start the main function
    main()
