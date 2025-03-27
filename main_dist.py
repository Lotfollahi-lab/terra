import os
import asyncio
import torch
import logging
import torch.distributed as dist
from src.nichejepa.train import train
from src.nichejepa.datasets.utils import prepare_dataset
from src.nichejepa.utils.config import create_params_from_YAML_wandb_config
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

    TMP_ARTIFACT_LOCATION = "/tmp"

    if WORLD_RANK==0:

        current_timestamp = (
                datetime.now().strftime("%d%m%Y_%H%M%S") +
                f"_{datetime.now().microsecond // 1000:03d}")
        logger.info(f'Run timestamp: {current_timestamp}.')
        logger.info(params)

        if params['state']['folder_path'] is None:
            # Define path for the dataset directory within the artifact location
            dataset_dir = os.path.join(TMP_ARTIFACT_LOCATION, params['data']['dataset_name'])
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir, exist_ok=True)
                logger.info(f"Created dataset directory: {dataset_dir}")
            # Define run folder with current timestamp inside the dataset directory
            folder_path = os.path.join(dataset_dir, current_timestamp)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
                logger.info(f"Created run directory: {folder_path}")
        else:
            folder_path = params['state']['folder_path']
    else:
        # For non-master processes, set folder_path to None or appropriate default
        folder_path = None

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
        timeout=timedelta(seconds=120)
    )

    # First, handle artifacts (for rank 0 only)
    if WORLD_RANK == 0:
        MY_ARTIFACT_LOCATION = os.path.join(os.environ.get('OUTPUT_DIR'), "artifacts")
        logger.info(f"My artifact location: {MY_ARTIFACT_LOCATION}")
        if not os.path.exists(MY_ARTIFACT_LOCATION):
            os.makedirs(MY_ARTIFACT_LOCATION, exist_ok=True)

    train_dataset, val_dataset, test_dataset = prepare_dataset(params)
    train(params,
          train_dataset,
          test_dataset,
          save_folder_path=folder_path,
          LOCAL_RANK=LOCAL_RANK,
          WORLD_SIZE=WORLD_SIZE,
          RANK=WORLD_RANK)


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
