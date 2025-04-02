import os
import asyncio
import torch
import logging
import torch.distributed as dist
from src.nichejepa.train import train
from src.nichejepa.datasets.utils import prepare_dataset
from src.nichejepa.utils.config import create_params_from_YAML_wandb_config
from src.nichejepa.utils.setup_experiment_folders import setup_folders
import wandb
import sys
# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import argparse
import warnings
from datetime import datetime
from datetime import timedelta


warnings.filterwarnings("ignore")
# logger
logging.basicConfig()
logger = logging.getLogger()

# ==========================



# Function to retrieve and log distributed environment variables
def get_distributed_info():
    """
    Retrieves distributed training environment variables and logs them.

    Returns
    -------
    tuple
        (WORLD_RANK, LOCAL_RANK, WORLD_SIZE)
    """
    if "LOCAL_RANK" in os.environ:
        # Environment variables set by torch.distributed.launch or torchrun
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


    # Handle artifacts (create directory only on rank 0, but share path with all ranks)
    output_dir = os.environ.get('OUTPUT_DIR')
    if not output_dir:
        raise ValueError("OUTPUT_DIR environment variable must be set")

    folder_path, experiment_artifact_location = setup_folders(
        world_rank=WORLD_RANK,
        tmp_artifact_path="/tmp",
        artifact_location=os.path.join(output_dir, "artifacts"),
        params=params,
        logger=logger
    )

    # if WORLD_RANK==0:
    #     wandb.init(project='nichejepa-sweep', id=run_id, resume="allow", group="multi_node_training", mode='online')

    print(f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")

    torch.cuda.set_device(LOCAL_RANK)

    # Initialize the distributed backend
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        rank=WORLD_RANK,
        world_size=WORLD_SIZE,
        timeout=timedelta(minutes=10)
    )

    # Wait for all processes to reach this point
    dist.barrier()

    train_dataset, val_dataset, test_dataset = prepare_dataset(params)
    train(params,
          train_dataset,
          test_dataset,
          save_folder_path=folder_path,
          my_artifact_location=experiment_artifact_location,
          local_rank=LOCAL_RANK,
          world_size=WORLD_SIZE,
          world_rank=WORLD_RANK)


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

    # Set additional environment variables
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    # Start the main function
    main()
