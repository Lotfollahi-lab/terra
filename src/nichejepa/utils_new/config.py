import yaml
import logging
from typing import Any

def create_params_from_YAML_wandb_config(YAML_file: str,
                                         logger: logging.RootLogger,
                                         sweep_config: Any = None,
                                         has_same_dimension: bool = True,
                                         update_from_sweep: bool = False
                                         ) -> dict:
    """
    Updates the `params` dictionary with values from the YAML config
    file and optionally from the wandb configuration file. This can be
    useful when using wandb sweeps for hyperparameter optimization.
    Also sets the seed in `params` from `args.seed`.

    Parameters
    -----------
    YAML_file:
        contains the filename of the YAML configuration for static
        params.
    logger:
        Logger object to log the loaded parameters.
    sweep_config:
        A configuration object (such as one from wandb) containing the
        parameters to update in `params` for dynamic change.
    has_same_dimension:
        Indicates if the pred_emb_dim and enc_emb_dim should be the same.
    update_from_sweep:
        Flag to determine whether to update parameters from the
        sweep_config (wandb).

    Returns
    -----------
    dict:
        The updated `params` dictionary.
    """
    try:
        # Load parameters from YAML configuration file
        with open(YAML_file, 'r') as y_file:
            params = yaml.safe_load(y_file)
            logger.info('Loaded parameters from YAML file.')
    except FileNotFoundError:
        logger.error(f"YAML configuration file '{YAML_file}' not found.")
        raise
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML file: {exc}")
        raise

    if update_from_sweep:
        # Update 'meta' section
        if hasattr(sweep_config, 'enc_pred_depth'):
            params['meta']['enc_pred_depth'] = int(sweep_config.enc_pred_depth)
            params['meta']['pred_depth'] = int(sweep_config.enc_pred_depth % 10)
            params['meta']['enc_depth'] = int(sweep_config.enc_pred_depth // 10)
        if hasattr(sweep_config, 'enc_emb_dim'):
            params['meta']['enc_emb_dim'] = sweep_config.enc_emb_dim
            if has_same_dimension:
                params['meta']['pred_emb_dim'] = sweep_config.enc_emb_dim

        # Update 'mask' section
        if hasattr(sweep_config, 'n_targets'):
             params['mask']['n_targets'] = sweep_config.n_targets
        if hasattr(sweep_config, 'per_block_mask_ratio'):
            params['mask']['per_block_mask_ratio'] = sweep_config.per_block_mask_ratio

        # Update 'optimization' section
        if hasattr(sweep_config, 'ema'):
            params['optimization']['ema'] = sweep_config.ema
        if hasattr(sweep_config, 'epochs'):
            params['optimization']['epochs'] = sweep_config.epochs

        print(params)

    # Return the updated params dictionary
    return params