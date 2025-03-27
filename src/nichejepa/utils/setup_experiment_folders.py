import logging
import os
from datetime import datetime

def setup_experiment_folders(
    world_rank: int,
    tmp_artifact_path: str,
    params: dict,
    logger: logging.Logger | None = None
    ) -> str | None:
    """Sets up experiment folders for distributed training.

    Creates necessary directories only on the master node (rank 0).

    Args
    ----
        world_rank (int): The rank of the current process in distributed training.
        tmp_artifact_location (str): Base directory for temporary artifacts.
        params (dict): Configuration parameters containing state and data information.
        logger (logging.Logger | None, optional): Logger instance for logging operations.
            Defaults to None.

    Returns
    -------
        str | None: Path to the experiment folder for rank 0, None for other ranks.

    Example
    -------
        >>> params = {
        ...     'state': {'folder_path': None},
        ...     'data': {'dataset_name': 'my_dataset'}
        ... }
        >>> folder_path = setup_experiment_folders(0, "/tmp", params)
    """
    if world_rank == 0:
        # Generate timestamp for unique folder naming
        current_timestamp = (
            datetime.now().strftime("%d%m%Y_%H%M%S") +
            f"_{datetime.now().microsecond // 1000:03d}"
        )

        if logger:
            logger.info(f'Run timestamp: {current_timestamp}.')
            logger.info(params)

        if params['state']['folder_path'] is None:
            # Define path for the dataset directory within the artifact location
            dataset_dir = os.path.join(tmp_artifact_path, params['data']['dataset_name'])
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir, exist_ok=True)
                if logger:
                    logger.info(f"Created dataset directory: {dataset_dir}")

            # Define run folder with current timestamp inside the dataset directory
            folder_path = os.path.join(dataset_dir, current_timestamp)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
        else:
            folder_path = params['state']['folder_path']

        return folder_path
    else:
        # For non-master processes, return None
        return None

def setup_artifact_location(
    world_rank: int,
    output_dir: str,
    logger: logging.Logger | None = None
) -> str | None:
    """Sets up artifact location for distributed training.

    Creates necessary directories only on the master node (rank 0).

    Args
    ----
        world_rank (int): The rank of the current process in distributed training.
        output_dir (str): Base directory for output artifacts.
        params (dict): Configuration parameters containing state and data information.
        logger (logging.Logger | None, optional): Logger instance for logging operations.
            Defaults to None.

    Returns
    -------
        str | None: Path to the artifact location for rank 0, None for other ranks.
    """
    if world_rank == 0:
        artifact_location = os.path.join(output_dir, "artifacts")
        if not os.path.exists(artifact_location):
            os.makedirs(artifact_location, exist_ok=True)
            if logger:
                logger.info(f"Created artifact directory: {artifact_location}")
        return artifact_location
    return None
