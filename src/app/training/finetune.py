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


@torch.no_grad()
def validate(
        model,
        val_loader,
        val_label_lookup,
        criterion,
        device,
        n_special_tokens,
        model_config,
        selection_type,
        agg_excluded_tokens,
        top_k
    ):
    """
    Run validation and return loss and accuracy.
    
    Parameters
    ----------
    model:
        The model to validate.
    val_loader:
        Validation dataloader.
    val_label_lookup:
        Label lookup dictionary for validation data.
    criterion:
        Loss function.
    device:
        Device to run validation on.
    n_special_tokens:
        Number of special tokens.
    model_config:
        Model configuration dictionary.
    selection_type:
        Selection type for cell mask creation.
    agg_excluded_tokens:
        Excluded tokens for aggregation.
    top_k:
        Top k for selection.
    
    Returns
    -------
    val_loss:
        Average validation loss.
    val_accuracy:
        Validation accuracy.
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    for udata, _, _, masks_attention in val_loader:
        for key in udata.keys():
            if key != 'cell_id':
                udata[key] = udata[key].to(device, non_blocking=True)
        masks_attention = masks_attention.to(device, non_blocking=True)
        
        ns_tokens = udata['tokens'][:, n_special_tokens:]
        cell_mask = create_binary_selection_mask(
            ns_tokens,
            selection_type=selection_type,
            excluded_tokens=agg_excluded_tokens,
            seq_len_cell=model_config['data']['seq_len_cell'],
            top_k=top_k,
        )
        
        logits = model(
            udata=udata,
            masks_attention=masks_attention,
            cell_mask=cell_mask,
        )
        
        labels = torch.tensor(
            val_label_lookup[udata['cell_id']].values,
            dtype=torch.long,
            device=device
        )
        
        loss = criterion(logits, labels)
        val_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1)
        val_correct += (preds == labels).sum().item()
        val_total += labels.size(0)
    
    model.train()
    
    return val_loss / len(val_loader), val_correct / val_total


def finetune(
        args: dict,
        train_adata: AnnData,
        train_dataset: Dataset,
        val_adata: AnnData | None = None,
        val_dataset: Dataset | None = None,
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
    train_adata:
        AnnData object containing labels for the finetune training dataset.
    train_dataset:
        Finetune training dataset.
    val_adata:
        AnnData object containing labels for the validation dataset (optional).
    val_dataset:
        Validation dataset (optional).
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
    train_loader, _ = init_dataloader_and_sampler(
        cell_dataset=train_cell_dataset,
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
    # PREPARE VALIDATION DATASET AND DATALOADER
    # -------------------------------------------------------------------- #
    val_loader = None
    val_label_lookup = None
    if val_dataset is not None and val_adata is not None:
        logger.info("Preparing validation dataset and dataloader...")
        
        # Create validation cell dataset
        val_cell_dataset = init_cell_dataset(
            dataset=val_dataset,
            vocab_size=vocab_size,
            seq_len_cell=model_config['data']['seq_len_cell'],
            seq_len_neighborhood=model_config['data']['seq_len_neighborhood'],
            tokenizer_type=model_config['data']['tokenizer_type'],
            gt_type=model_config['meta']['gt_type'],
            cell_pos_enc=model_config['meta']['cell_pos_enc'],
            special_tokens=model_config['meta']['special_tokens'],
            sampling_strategy=None,
            n_nonzero_tokens_list=None,
            include_cell_id=True,
            sep_gene_tokens_neb=model_config['data']['sep_gene_tokens_neb'])
        
        # Initialize validation dataloader
        val_loader, _ = init_dataloader_and_sampler(
            cell_dataset=val_cell_dataset,
            batch_size=args['data']['batch_size'],
            distributed=False,
            world_size=1,
            rank=0,
            collate_fn=mask_collator,
            pin_memory=args['data']['pin_memory'],
            num_workers=args['data']['num_workers'],
            drop_last=False,
            persistent_workers=False)
        
        logger.info(f"Validation dataset size: {len(val_dataset)} samples")
    else:
        logger.info("No validation dataset provided. Skipping validation.")

    # -------------------------------------------------------------------- #
    # PREPARE LABELS
    # -------------------------------------------------------------------- #
    logger.info("Preparing labels...")
    label_name = args['data']['label_name']
    
    # Assert required columns exist in train_adata.obs
    assert 'cell_id' in train_adata.obs.columns, f"'cell_id' not found in train_adata.obs.columns. Available columns: {train_adata.obs.columns.tolist()}"
    assert label_name in train_adata.obs.columns, f"'{label_name}' not found in train_adata.obs.columns. Available columns: {train_adata.obs.columns.tolist()}"
    
    # Assert that label_name contains integer values (required for CrossEntropyLoss)
    label_dtype = train_adata.obs[label_name].dtype
    assert pd.api.types.is_integer_dtype(label_dtype), (
        f"Labels in '{label_name}' must be integers, but got dtype: {label_dtype}. "
        f"Please encode string labels to integer class indices."
    )
    
    # Create label lookup dictionary indexed by cell_id (do once before training loop)
    label_lookup = train_adata.obs.set_index('cell_id')[label_name]
    logger.info(f"Label lookup: {label_lookup}")

    # Create validation label lookup if validation data is provided
    if val_adata is not None:
        assert 'cell_id' in val_adata.obs.columns, f"'cell_id' not found in val_adata.obs.columns."
        assert label_name in val_adata.obs.columns, f"'{label_name}' not found in val_adata.obs.columns."
        val_label_dtype = val_adata.obs[label_name].dtype
        assert pd.api.types.is_integer_dtype(val_label_dtype), (
            f"Validation labels in '{label_name}' must be integers, but got dtype: {val_label_dtype}."
        )
        val_label_lookup = val_adata.obs.set_index('cell_id')[label_name]
        logger.info(f"Validation label lookup created with {len(val_label_lookup)} entries")

    # set number of classes
    try:
        num_classes = train_adata.uns[f'{label_name}_num_classes']
    except KeyError:
        num_classes = train_adata.obs[label_name].nunique()
        logger.warning(f"Number of classes not found in train_adata.uns. Using train_adata.obs[{label_name}].nunique() instead.")
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

    # apply PEFT-adapter if specified
    if use_peft:
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

    use_mlp = args['finetune']['use_mlp']
    hidden_dim = args['finetune']['hidden_dim']
    lr = args['finetune']['lr']
    num_epochs = args['finetune']['num_epochs']
    selection_type = args['finetune']['selection_type']
    agg_excluded_tokens = args['finetune']['excluded_tokens']
    top_k = args['finetune']['top_k']
    
    # apply a linear layer or MLP to the output of the PEFT target encoder if PEFT is applied, otherwise apply to the output of the target encoder
    model = ClassificationModel(
        base_model=peft_target_encoder if use_peft else target_encoder,
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
    logger.info(f"Model name: {model_name}")
    
    # Save checkpoint function
    def save_checkpoint(
            epoch,
            train_loss,
            train_accuracy,
            val_loss=None,
            val_accuracy=None,
        ):
        """
        Save checkpoint.
        Parameters
        ----------
        epoch:
            Epoch number.
        train_loss:
            Training loss for the epoch.
        train_accuracy:
            Training accuracy for the epoch.
        val_loss:
            Validation loss for the epoch (optional).
        val_accuracy:
            Validation accuracy for the epoch (optional).
        """
        save_dict = {
                        'model': model.state_dict(),
                        'opt': optimizer.state_dict(),
                        'epoch': epoch,
                        'zero_epoch_tracking': True,
                        'train_loss': train_loss,
                        'train_accuracy': train_accuracy,
                        'val_loss': val_loss,
                        'val_accuracy': val_accuracy,
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
    # Track best validation metrics
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch}")
        model.train()  # Ensure training mode
        train_running_loss = 0.0
        train_correct_preds = 0
        train_total_preds = 0

        for itr, (udata, _, _, masks_attention) in tqdm(enumerate(train_loader)):
            for key in udata.keys():
                if key != 'cell_id':
                    udata[key] = udata[key].to(device, non_blocking=True)
            masks_attention = masks_attention.to(device, non_blocking=True)
            
            ns_tokens = udata['tokens'][:, n_special_tokens:]

            # Aggregate gene embeddings into cell and neighborhood embeddings
            cell_mask = create_binary_selection_mask(
                ns_tokens,
                selection_type=selection_type,
                excluded_tokens=agg_excluded_tokens, # None
                seq_len_cell=model_config['data']['seq_len_cell'],
                top_k=top_k, # None
            )
            
            optimizer.zero_grad()

            # Forward pass
            logits = model(
                udata=udata,
                masks_attention=masks_attention,
                cell_mask=cell_mask,
            )

            # Get labels for this batch by matching cell IDs
            # udata['cell_id'] is a list of cell IDs
            labels = torch.tensor(
                label_lookup[udata['cell_id']].values,
                dtype=torch.long,
                device=device
            )

            # Compute the loss
            train_loss_batch = criterion(
                input=logits,
                target=labels
            )

            # Backward pass and optimization
            train_loss_batch.backward()
            optimizer.step()

            # Track statistics
            train_running_loss += train_loss_batch.item()
            
            # Track accuracy
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                train_correct_preds += (preds == labels).sum().item()
                train_total_preds += labels.size(0)
                
            # if itr > 5:
                # break

        train_loss = train_running_loss / len(train_loader)
        train_accuracy = train_correct_preds / train_total_preds
        
        # Validation
        val_loss = None
        val_accuracy = None
        if val_loader is not None:
            val_loss, val_accuracy = validate(
                model=model,
                val_loader=val_loader,
                val_label_lookup=val_label_lookup,
                criterion=criterion,
                device=device,
                n_special_tokens=n_special_tokens,
                model_config=model_config,
                selection_type=selection_type,
                agg_excluded_tokens=agg_excluded_tokens,
                top_k=top_k
            )
            
            # Track best validation metrics
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
        
        # Logging
        if val_loss is not None:
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )
        else:
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
            )

        save_checkpoint(epoch, train_loss, train_accuracy, val_loss, val_accuracy)
    
    if val_loader is not None:
        logger.info(f"Best validation loss: {best_val_loss:.4f}, Best validation accuracy: {best_val_accuracy:.4f}")
    logger.info(f"Finetuning completed.")


if __name__ == '__main__':
    
    # load args dictionary from config file
    args = parse_arguments()
    
    # load finetune dataset (tokenized)
    train_dataset = load_from_disk(args['data']['finetune_dataset'])
    cols = [c for c in train_dataset.column_names]
    train_dataset.set_format(
        type="torch",
        columns=cols,
        output_all_columns=True
    )
    logger.info(f"Finetune training dataset sample: {train_dataset[0]}")

    # load finetune adata (labels)
    train_adata = sc.read_h5ad(args['data']['finetune_adata'])
    logger.info(f"Finetune training adata: {train_adata}")

    # Load validation data (optional)
    val_dataset = None
    val_adata = None
    if 'val_dataset' in args['data'] and 'val_adata' in args['data']:
        val_dataset = load_from_disk(args['data']['val_dataset'])
        val_dataset.set_format(
            type="torch",
            columns=cols,
            output_all_columns=True
        )
        val_adata = sc.read_h5ad(args['data']['val_adata'])
        logger.info(f"Validation dataset loaded: {len(val_dataset)} samples")
        logger.info(f"Validation adata: {val_adata}")
    else:
        logger.info("No validation dataset specified in config. Training without validation.")

    # finetune model
    finetune(
        args=args,
        train_dataset=train_dataset,
        train_adata=train_adata,
        val_dataset=val_dataset,
        val_adata=val_adata,
    )