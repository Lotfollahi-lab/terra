import os
import logging
from datetime import datetime
from typing import Tuple, Optional

def setup_folders(
    world_rank: int,
    tmp_artifact_path: str,
    artifact_location: str,
    params: dict,
    logger: Optional[logging.Logger] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Sets up experiment and artifact folders for distributed training.

    For the master node (world_rank == 0):
      - Ensures the base artifact directory exists.
      - If `params['state']['folder_path']` is None:
          - Generates a unique timestamp.
          - Creates a dataset directory under `tmp_artifact_path` using `params['data']['dataset_name']`.
          - Creates a timestamped experiment folder under that dataset directory.
      - Otherwise, uses the provided experiment folder in `params['state']['folder_path']`
        (and extracts the timestamp from the folder name).
      - In both cases, creates an artifact experiment folder at:
          os.path.join(artifact_location, dataset_name, current_timestamp)

    Args:
        world_rank (int): The rank of the current process in distributed training.
        tmp_artifact_path (str): Base directory for temporary artifacts.
        artifact_location (str): Base artifact directory (typically os.path.join(output_dir, "artifacts")).
        params (dict): Configuration parameters that must include:
            - 'data': {'dataset_name': ...}
            - 'state': {'folder_path': ...} (can be None)
        logger (logging.Logger | None, optional): Logger instance for logging operations.

    Returns
    -------
        Tuple[Optional[str], Optional[str]]:
            - The experiment folder path under tmp_artifact_path.
            - The corresponding artifact experiment folder path.
            Returns (None, None) for non-master nodes.

    Example:
        >>> params = {
        ...     'state': {'folder_path': None},
        ...     'data': {'dataset_name': 'my_dataset'}
        ... }
        >>> exp_folder, artifact_exp_folder = setup_folders(
        ...     world_rank=0,
        ...     tmp_artifact_path="/tmp",
        ...     artifact_location=os.path.join("/output", "artifacts"),
        ...     params=params,
        ...     logger=my_logger
        ... )
    """
    if world_rank == 0:
        # Ensure the base artifact directory exists
        if not os.path.exists(artifact_location):
            os.makedirs(artifact_location, exist_ok=True)
            if logger:
                logger.info(f"Created artifact directory: {artifact_location}")

        # Extract the dataset name from params
        dataset_name = params.get('data', {}).get('dataset_name')
        if not dataset_name:
            raise ValueError("Missing 'dataset_name' in params['data']")

        # Determine experiment folder and timestamp
        if params.get('state', {}).get('folder_path') is None:
            # Generate a unique timestamp for the folder naming
            current_timestamp = (
                datetime.now().strftime("%d%m%Y_%H%M%S") +
                f"_{datetime.now().microsecond // 1000:03d}"
            )
            if logger:
                logger.info(f"Run timestamp: {current_timestamp}.")

            # Create dataset directory under tmp_artifact_path
            dataset_dir = os.path.join(tmp_artifact_path, dataset_name)
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir, exist_ok=True)
                if logger:
                    logger.info(f"Created dataset directory: {dataset_dir}")

            # Create the experiment folder under the dataset directory
            exp_folder = os.path.join(dataset_dir, current_timestamp)
            if not os.path.exists(exp_folder):
                os.makedirs(exp_folder, exist_ok=True)
                if logger:
                    logger.info(f"Created experiment folder: {exp_folder}")
        else:
            # Use the provided experiment folder
            exp_folder = params['state']['folder_path']
            current_timestamp = os.path.basename(exp_folder)

        # Create artifact experiment folder: artifact_location/dataset_name/current_timestamp
        artifact_dataset_dir = os.path.join(artifact_location, dataset_name)
        if not os.path.exists(artifact_dataset_dir):
            os.makedirs(artifact_dataset_dir, exist_ok=True)
            if logger:
                logger.info(f"Created artifact dataset directory: {artifact_dataset_dir}")
        artifact_exp_folder = os.path.join(artifact_dataset_dir, current_timestamp)
        if not os.path.exists(artifact_exp_folder):
            os.makedirs(artifact_exp_folder, exist_ok=True)
            if logger:
                logger.info(f"Created artifact experiment folder: {artifact_exp_folder}")

        return exp_folder, artifact_exp_folder
    else:
        # For non-master nodes, return None for both folders
        return None, None
