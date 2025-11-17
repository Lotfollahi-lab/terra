"""
Usage: python source-code/src/app/finetune.py --fname reproducibility/config/finetuning/xhs1000_niche_finetune-nemo.yaml
"""
import os
import sys
import argparse
import pickle
import logging
import yaml
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from app.utils import (init_model, load_checkpoint, parse_arch_kwargs,
                       parse_protein_init_kwargs)
from terra.datasets.cell_datasets import CellBaseDataset
from terra.datasets.dataloaders import init_dataloader_and_sampler
from terra.models.modules import ClassificationModel


os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1" # Better error propagation

_GLOBAL_SEED = 0
LOCAL_RANK = None

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


# Setup argument parsing
def parse_arguments():
    """
    Parse config file name from command-line arguments and return hyperparameters in a nested dictionary.
    """
    parser = argparse.ArgumentParser(
        description='Run NicheJEPA finetuning.')
    parser.add_argument('--fname', type=str, default='configs.yaml',
                        help='Name of the config file to load.')
    
    parser_args = parser.parse_args()
    
    # Get the config file name from command line argument
    args_fname = parser_args.fname

    # Read parameters from config file
    with open(args_fname, "r") as f:
        args = yaml.safe_load(f)    

    return args


# @torch.no_grad()
def finetune(
        args: dict,
        adata: AnnData,
        dataset: Dataset,
        save_folder_path: str | None = None,
        LOCAL_RANK: int | None = None,
        WORLD_RANK: int | None = None,
    ):
    """
    Train model.

    Parameters
    -----------
    args:
        Dictionary containing the hyperparams from the config file.
    adata:
        AnnData object containing labels for the finetune training dataset.
    dataset:
        Finetune training dataset.
    emb_layer:
        Layer for which to retrieve the embedding.
    save_folder_path:
        Path for saving model artifacts.
    LOCAL_RANK:
        Local rank of the process.
    WORLD_RANK:
        World rank of the process.
    """
    # -------------------------------------------------------------------- #
    # BACKEND SETUP
    # -------------------------------------------------------------------- #
    # Set random seeds
    logger.info("Configuring backend...")
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_GLOBAL_SEED)
    torch.backends.cudnn.deterministic = False # set to True for reproducibility
    torch.backends.cudnn.benchmark = True # set to False for reproducibility

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    elif LOCAL_RANK is not None:
        device = torch.device(f"cuda:{LOCAL_RANK}")
    elif LOCAL_RANK is None:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -------------------------------------------------------------------- #
    # LOAD MODEL CONFIG
    # -------------------------------------------------------------------- #
    logger.info("Loading model config...")
    # Construct paths to model config, token dictionary, and model checkpoint
    pretrained_checkpoint_path = Path(args['model']['pretrained_checkpoint_path'])
    model_config_file_path = pretrained_checkpoint_path / 'model_config.yaml'
    token_dictionary_file_path = pretrained_checkpoint_path / 'token_dictionary.pkl'
    model_checkpoint_path = pretrained_checkpoint_path / 'model_checkpoint.pt'

    # Load model config
    with open(model_config_file_path, 'r') as file:
        model_config = yaml.safe_load(file)

    # -------------------------------------------------------------------- #
    # LOAD TOKEN DICTIONARY AND GET TOKEN DICTIONARY-SPECIFIC PARAMS
    # -------------------------------------------------------------------- #
    logger.info("Loading token dictionary...")
    # Get token sequence length and number of special tokens
    n_special_tokens = len(model_config['meta']['special_tokens'])
    seq_len = (
        model_config['data']['seq_len_cell'] +
        model_config['data']['seq_len_neighborhood'] +
        n_special_tokens)

    # Load token dict and get token dict-specfic params
    with open(token_dictionary_file_path, 'rb') as file:
        token_dict = pickle.load(file)
    vocab_size = len(token_dict)
    n_special_values = sum(1 for key in token_dict if "spv" in key)

    # Mirror train.py: if the original training run used protein
    # initialization for the token embedding, the encoder must be
    # rebuilt with the same structure or state_dict loading will fail.
    protein_init_kwargs = parse_protein_init_kwargs(args)

    n_contexts = args['mask']['n_contexts']
    n_targets = args['mask']['n_targets']
    block_masking = args['mask']['block_masking']
    cell_masking = args['mask']['cell_masking']
    context_mask_size = args['mask']['context_mask_size']
    target_mask_size = args['mask']['target_mask_size']
    per_block_mask_ratio = args['mask']['per_block_mask_ratio']
    if 'sample_segments' in args['mask'].keys():
        sample_segments = args['mask']['sample_segments']
    else:
        sample_segments = False
    targets_list = args['mask']['targets_list']

    # Initialize dataloader
    loader, _ = init_dataloader_and_sampler(
        cell_dataset=cell_dataset,
        batch_size=args['data']['batch_size'],
        distributed=False,
        world_size=1,
        rank=0,
        collate_fn=mask_collator,
        pin_memory=args['data']['pin_memory'],
        num_workers=args['data']['num_workers'],
        drop_last=False,
        persistent_workers=False)

    # -------------------------------------------------------------------- #
    # PREPARE LABELS
    # -------------------------------------------------------------------- #
    logger.info("Preparing labels...")
    label_name = args['data']['label_name']
    
    # Assert required columns exist in adata.obs
    assert 'cell_id' in adata.obs.columns, f"'cell_id' not found in adata.obs.columns. Available columns: {adata.obs.columns.tolist()}"
    assert label_name in adata.obs.columns, f"'{label_name}' not found in adata.obs.columns. Available columns: {adata.obs.columns.tolist()}"
    
    # Assert that label_name contains integer values (required for CrossEntropyLoss)
    label_dtype = adata.obs[label_name].dtype
    assert pd.api.types.is_integer_dtype(label_dtype), (
        f"Labels in '{label_name}' must be integers, but got dtype: {label_dtype}. "
        f"Please encode string labels to integer class indices."
    )
    
    # Create label lookup dictionary indexed by cell_id (do once before training loop)
    label_lookup = adata.obs.set_index('cell_id')[label_name]
    logger.info(f"Label lookup: {label_lookup}")

    # set number of classes
    try:
        num_classes = adata.uns[f'{label_name}_num_classes']
    except KeyError:
        num_classes = adata.obs[label_name].nunique()
        logger.warning(f"Number of classes not found in adata.uns. Using adata.obs[{label_name}].nunique() instead.")
    logger.info(f"Number of classes: {num_classes}")

    # Load token dict and get token dict-specfic params
    with open(token_dict_folder_path, 'rb') as file:
        token_dict = pickle.load(file)
    vocab_size = len(token_dict)
    if protein_init_kwargs is not None:
        protein_init_kwargs['token_dict'] = token_dict
    n_special_values = sum(
        1 for key in token_dict if "spv" in key) # this only works now because of the dummy special values
    max_special_tokens = sum(
        1 for key in token_dict if "cls" in key) + sum(
        1 for key in token_dict if "spt" in key)

    # Define tokenizer-specific params
    if tokenizer_type == 'cell_neighborhood':
        if add_cls:
            special_tokens = ['cls_0', 'cls_1'] + special_tokens            
    elif tokenizer_type == 'cell_graph':
        if add_cls:
            special_tokens = [
                f'cls_{i}' for i in range(n_segments)] + special_tokens

    # Get token sequence length and number of special tokens
    n_special_tokens = len(special_tokens)
    seq_len = seq_len_cell + seq_len_neighborhood + n_special_tokens

    # Set multiprocessing start method
    try:
        mp.set_start_method("spawn")
    except Exception:
        logger.info(f'Multiprocessing not started.')

    # Initialize torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}.')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # Create folder to store artifacts
    if not save_folder_path:
        finetune_checkpoint_path = Path(args['model']['finetune_checkpoint_path'])
    else:
        finetune_checkpoint_path = save_folder_path
    current_timestamp = (
                datetime.now().strftime("%d%m%Y_%H%M%S") +
                f"_{datetime.now().microsecond // 1000:03d}")
    finetune_dir = finetune_checkpoint_path / current_timestamp
    finetune_dir.mkdir(parents=True, exist_ok=True)
    
    with open(finetune_dir / 'params.yaml', 'w') as f:
        yaml.dump(args, f)

    # Pull architecture hyperparameters that must round-trip from
    # the saved config (laplacian / rope / adaln). Without this the
    # checkpoint load below would silently use the init_model defaults
    # for these knobs, producing the wrong architecture or runtime
    # behavior versus how it was trained.
    arch_kwargs = parse_arch_kwargs(args)

    # Initialize target encoder
    target_encoder, _ = init_model(
        gt_type=model_config['meta']['gt_type'],
        count_encoding=model_config['meta']['count_encoding'],
        n_value_bins=model_config['meta']['n_value_bins'],
        cell_pos_enc=model_config['meta']['cell_pos_enc'],
        device=device,
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_special_tokens=n_special_tokens,
        n_segments=model_config['data']['n_segments'],
        n_special_values=n_special_values,
        enc_emb_dim=enc_emb_dim,
        enc_depth=enc_depth,
        pred_emb_dim=pred_emb_dim,
        pred_depth=pred_depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        use_flash_attention=use_flash_attention,
        use_layer_norm=use_layer_norm,
        sep_gene_tokens_neb=sep_gene_tokens_neb,
        protein_init_kwargs=protein_init_kwargs,
        **arch_kwargs)

    # if api_version != 'v3':
    #     return_layer_emb_fn = target_encoder.return_layer_emb
    # else:
    #     return_layer_emb_fn = target_encoder.backbone.return_layer_emb

    # Load model checkpoint
    _, _, target_encoder, _, _, _, _ = load_checkpoint(
            device=device,
            r_path=model_checkpoint_path,
            encoder=None,
            predictor=None,
            target_encoder=target_encoder,
            opt=None,
            scaler=None,
            is_training=False)
    
    # TODO: Incorporate DistributedDataParallel here if we want to finetune the model in a distributed manner.
    
    # -------------------------------------------------------------------- #
    # BUILD MODEL
    # -------------------------------------------------------------------- #
    # get finetune hyperparameters
    use_peft = args['finetune']['use_peft']
    peft_method = args['finetune']['peft_method']
    peft_rank = args['finetune']['peft_rank']
    peft_alpha = args['finetune']['peft_alpha']
    peft_dropout = args['finetune']['peft_dropout']
    peft_bias = args['finetune']['peft_bias']
    peft_task_type = args['finetune']['peft_task_type']
    try:
        peft_target_modules = args['finetune']['peft_target_modules']
    except KeyError:
        peft_target_modules = None

    use_mlp = args['finetune']['use_mlp']
    hidden_dim = args['finetune']['hidden_dim']
    lr = args['finetune']['lr']
    num_epochs = args['finetune']['num_epochs']
    selection_type = args['finetune']['selection_type']
    agg_type = args['finetune']['agg_type']
    agg_excluded_tokens = args['finetune']['excluded_tokens']
    top_k = args['finetune']['top_k']
    
    # apply PEFT-adapter if specified
    if use_peft:
        peft_target_encoder = apply_peft(
            target_encoder=target_encoder,
            peft_method=peft_method,
            peft_rank=peft_rank,
            peft_alpha=peft_alpha,
            peft_dropout=peft_dropout,
            peft_bias=peft_bias,
            peft_target_modules=peft_target_modules,
            peft_task_type=peft_task_type
        )
    else:
        # freeze parameters of target encoder
        for p in target_encoder.parameters():
            p.requires_grad = False
        logger.info(f"Target encoder parameters are frozen.")

    # apply a linear layer or MLP to the output of the PEFT target encoder if PEFT is applied, otherwise apply to the output of the target encoder
    model = ClassificationModel(
        base_model=peft_target_encoder if use_peft else target_encoder,
        agg_type=agg_type,
        num_classes=num_classes,
        use_mlp=use_mlp,
        hidden_dim=hidden_dim
    )
    model.to(device)
    
    # -------------------------------------------------------------------- #
    # PREPARE TRAINING INGREDIENTS
    # -------------------------------------------------------------------- #
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer (only optimize PEFT parameters)
    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )
    
    # -------------------------------------------------------------------- #
    # SAVE CHECKPOINT FUNCTION
    # -------------------------------------------------------------------- #
    model_name = "Zeroshot" if not use_peft else "Finetune"
    model_name += "+Linear" if not use_mlp else "MLP"
    
    # Save checkpoint function
    def save_checkpoint(
            epoch,
            epoch_loss,
            epoch_accuracy,
        ):
        """
        Save checkpoint.
        Parameters
        ----------
        epoch:
            Epoch number.
        epoch_loss:
            Loss for the epoch.
        epoch_accuracy:
            Accuracy for the epoch.
        """
        save_dict = {
                        'model': model.state_dict(),
                        'opt': optimizer.state_dict(),
                        'epoch': epoch,
                        'zero_epoch_tracking': True,
                        'loss': epoch_loss,
                        'accuracy': epoch_accuracy,
                        'batch_size': args['data']['batch_size'],
                        'world_size': 1,
                        'lr': lr,
                        'model_name': model_name,
                    }
        ft_model_name = finetune_dir / f'ft_{epoch}.pt'
        logger.info(f"Saving checkpoint to {ft_model_name}...")
        torch.save(
            save_dict,
            ft_model_name
        )

    # -------------------------------------------------------------------- #
    # TRAINING LOOP
    # -------------------------------------------------------------------- #
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch}")
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for _, (udata, _, _, masks_attention) in tqdm(enumerate(loader)):
            for key in udata.keys():
                if key != 'cell_id':
                    udata[key] = udata[key].to(device, non_blocking=True)
            masks_attention = masks_attention.to(device, non_blocking=True)
            
            ns_tokens = udata['tokens'][:, n_special_tokens:]

            # Aggregate gene embeddings into cell and neighborhood embeddings
            cell_mask = create_binary_selection_mask(
                ns_tokens,
                selection_type=selection_type,
                excluded_tokens=agg_excluded_tokens,
                seq_len_cell=model_config['data']['seq_len_cell'],
                top_k=top_k).to(device)
            if selection_type == 'agg_graph':
                neighborhood_mask = create_binary_selection_mask(
                    ns_tokens,
                    selection_type=selection_type,
                    excluded_tokens=agg_excluded_tokens,
                    seq_len_cell=model_config['data']['seq_len_cell'],
                    top_k=top_k,
                    n_segments=model_config['data']['n_segments']).to(device)
            else:
                neighborhood_mask = None
                    
            optimizer.zero_grad()

            # Forward pass
            logits = model(
                udata=udata,
                masks_attention=masks_attention,
                cell_mask=cell_mask,
                neighborhood_mask=neighborhood_mask
            )

            # Get labels for this batch by matching cell IDs
            # udata['cell_id'] is a list of cell IDs
            labels = torch.tensor(
                label_lookup[udata['cell_id']].values,
                dtype=torch.long,
                device=device
            )

            # Compute the loss
            loss = criterion(
                input=logits,
                target=labels
            )

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item()
            
            # Track accuracy
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                correct_preds += (preds == labels).sum().item()
                total_preds += labels.size(0)
                
            # break

        epoch_loss = running_loss / len(loader)
        epoch_accuracy = correct_preds / total_preds

        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        save_checkpoint(epoch, epoch_loss, epoch_accuracy)
        
    logger.info(f"Finetuning completed.")


if __name__ == '__main__':
    
    # load args dictionary from config file
    args = parse_arguments()
    
    # load finetune dataset (tokenized)
    dataset = load_from_disk(args['data']['finetune_dataset'])
    cols = [c for c in dataset.column_names]
    dataset.set_format(
        type="torch",
        columns=cols,
        output_all_columns=True
    )
    logger.info(f"Finetune training dataset sample: {dataset[0]}")

    # load finetune adata (labels)
    adata = sc.read_h5ad(args['data']['finetune_adata'])
    logger.info(f"Finetune training adata: {adata}")

    # finetune model
    finetune(
        args=args,
        dataset=dataset,
        adata=adata,
    )