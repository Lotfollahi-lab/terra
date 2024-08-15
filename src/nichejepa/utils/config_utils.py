import yaml
import logging
import pprint

def create_params_from_YAML_wandb_config(config, args, logger):
    """
    Updates the params dictionary with values from the config file and the object loaded from the wandb configuration file.
    This can be useful when we want to use sweeps with wandb for hyperparameter optimization.
    Also sets the seed in `params` from `args.seed`.

    Parameters:
    - config (object): A configuration object (such as one from wandb) containing the parameters to update in `params`.
    - args (object): An object that contains the filename of the YAML configuration and the seed value.
    - logger (object): log the loaded param
    Returns:
    - dict: The updated `params` dictionary.
    """

    # Load parameters from YAML configuration file
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('Loaded parameters:')
    
    # Update 'meta' section with values coming from config file in wandb
    params['meta']['pred_enc_depth'] = int(config.pred_enc_depth)
   # Extract the last digit of `pred_enc_depth` and assign it to `pred_depth`
    params['meta']['pred_depth'] = int(config.pred_enc_depth % 10)
    
    # Extract all digits except the last from `pred_enc_depth` and assign to `enc_depth`
    params['meta']['enc_depth'] = int(config.pred_enc_depth // 10)
    
    # Update embedding dimension for encoder and top layer configuration
    params['meta']['enc_emb_dim'] = config.enc_emb_dim
    params['meta']['top_layer'] = config.top_layer
    
    # Update the top-k parameter
    params['meta']['top_k'] = config.top_k
    
    # Update 'mask' section
    # Set the number of targets and context mask size from config
    params['mask']['n_targets'] = config.n_targets
    params['mask']['context_mask_size'] = config.context_mask_size
    
    # Update 'optimization' section
    # Set the EMA (Exponential Moving Average) parameter
    # Set the epochs

    params['optimization']['ema'] = config.ema
    params['optimization']['epochs'] = config.epochs
    params['optimization']['learnable'] = config.learnable

    # Set seed
    params['seed'] = args.seed

    # Return the updated params dictionary
    return params

def setup_batch_size(config):
    """
    Determine and set the appropriate batch size based on the model configuration.

    Parameters:
    - config (object): Configuration object with attributes `pred_enc_depth` and `epochs`.

    Returns:
    - int: The computed batch size.
    """

    # Default batch size assignment based on the encoder depth
    if config.pred_enc_depth < 41:
        batch_size = 80
    elif 41 <= config.pred_enc_depth < 51:
        batch_size = 40
    else:
        batch_size = 70

    # Adjust batch size if training for zero epochs (often used in testing or fast iterations)
    if config.epochs == 0:
        batch_size = 1000

    return batch_size

