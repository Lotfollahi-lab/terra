import logging
import yaml
from typing import Literal

import anndata as ad
import torch
from tqdm import tqdm

from app.helper import init_model, load_checkpoint
from nichejepa.datasets.cell_datasets import CellBaseDataset
from nichejepa.datasets.dataloaders import init_dataloader_and_sampler
from nichejepa.models.modules import ClassificationModel


@torch.no_grad()
def finetune(args: dict,
             dataset: CellBaseDataset,
                load_folder_path: str,
                dataset_ids: list | None = None,
                obs_cols: list | None = None,
                uns_cols: list | None = None,
                emb_layers: list | None = None,
                cell_gene_ids: list = [],
                neighborhood_gene_ids: list = [],
                agg_type: Literal['cls',
                                    'avg',
                                    'weighted_avg'] = 'avg',
                masked_tokens: list[int] | None = None,
                agg_excluded_tokens: list[int] | None = None,
                feature_norm: bool = False,
                use_peft: bool = False,
                ) -> ad.AnnData:
    """
    Use a trained model for inference. Run forward pass on a given
    dataset andbreturn cell, neighborhood and (optionally) gene
    embeddings (cell and neighborhood gene embeddings).

    Parameters
    -----------
    args:
        Dictionary containing the hyperparameters from the config file.
    dataset:
        Cell dataset for which embeddings will be inferred.
    load_folder_path:
        Path where the checkpoint is stored.
    emb_layers:
        Layers for which to retrieve the embedding.
    cell_gene_ids:
        List with gene IDs for which cell gene embeddings will be
        retrieved.
    neighborhood_gene_ids:
        List with gene IDs for which neighborhood gene embeddings will
        be retrived.
    agg_type:
        Specifies how (aggregated) cell and neighborhood embeddings are
        computed from individual gene embeddings.
    masked_tokens:
        List of tokens to be masked by the attention mask during
        inference.
    agg_excluded_tokens:
        List of tokens to be excluded from the aggregation.
    feature_norm:
        If `True`, apply feature norm in the last embedding layer.
    top_k:
        Include only top_k genes in aggregation.
    return_gene: 
        If 'True' will return gene_embedding.
    return_cosine_sim: 
        If 'True' will compute and return cosine_sim matrix.
    compute_cosine_with:
       If set to 'neighborhood', it will compute the cosine similarity between each cell and its neighborhood. 
       If set to 'cell', it will compute the cosine similarity between cells itself.
    Returns
    -----------
    adata:
        An AnnData object with the stored embeddings and labels.
    """
    # Set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # Load params from config file
    add_cls = args['meta']['add_cls']
    gt_type = args['meta']['gt_type']
    if 'count_encoding' in args['meta'].keys():
        count_encoding = args['meta']['count_encoding']
    else:
        count_encoding = 'value_bins'
    if 'n_value_bins' in args['meta'].keys():
        n_value_bins = args['meta']['n_value_bins']
    else:
        n_value_bins = 100
    enc_depth = args['meta']['enc_depth']
    enc_emb_dim = args['meta']['enc_emb_dim']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if 'num_heads' in args['meta'].keys():
        num_heads = args['meta']['num_heads']
    else:
        num_heads = 8
    if 'mlp_ratio' in args['meta'].keys():
        mlp_ratio = args['meta']['mlp_ratio']
    else:
        mlp_ratio = 4.0
    special_tokens = args['meta']['special_tokens']
    use_bfloat16 = args['meta']['use_bfloat16']
    use_flash_attention = args['meta']['use_flash_attention']
    
    if 'api_version' in args['meta'].keys():
        api_version = args['meta']['api_version']
    else:
        api_version = 'v3'

    dataset_name = args['data']['dataset_name']
    token_dict_folder_path = args['data']['token_dict_folder_path']
    raw_data_folder_path = args['data']['raw_data_folder_path']
    batch_size = args['data']['batch_size']
    pin_memory = args['data']['pin_memory']
    num_workers = args['data']['num_workers']
    tokenizer_type = args['data']['tokenizer_type']
    seq_len_cell = args['data']['seq_len_cell']
    seq_len_neighborhood = args['data']['seq_len_neighborhood']
    n_segments = args['data']['n_segments']
    MAX_OCC = args['data']['n_segments'] -1 

    n_contexts = args['mask']['n_contexts']
    n_targets = args['mask']['n_targets']
    block_masking = args['mask']['block_masking']
    if 'cell_masking' in args['mask'].keys():
        cell_masking = args['mask']['cell_masking']
    else:
        cell_masking = False
    context_mask_size = args['mask']['context_mask_size']
    target_mask_size = args['mask']['target_mask_size']
    per_block_mask_ratio = args['mask']['per_block_mask_ratio']
    if 'targets_list' in args['mask'].keys():
        targets_list = args['mask']['targets_list']
    else:
        targets_list = []

    r_file = args['state']['read_checkpoint']
    tag = args['state']['write_tag']

    if args['data']['precomputed_n_nonzero_tokens']:
        with open(args['data']['precomputed_n_nonzero_tokens'], "rb") as f: 
            n_nonzero_tokens = pickle.load(f)
    else:
        n_nonzero_tokens = None
    
    # Load token dict and get token dict-specfic params
    with open(token_dict_folder_path, 'rb') as file:
        token_dict = pickle.load(file)
    vocab_size = len(token_dict)
    n_special_values = sum(1 for key in token_dict if "spv" in key)
    max_special_tokens = sum(1 for key in token_dict if "cls" in key) + sum(
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

    # Specify last emb layer if not defined
    if emb_layers is None:
        emb_layers = [enc_depth]

    # Set the folder for saving extracted features
    save_folder = f"{load_folder_path}/extracted_features"
    feature_path = f"{save_folder}/"

    os.makedirs(save_folder, exist_ok=True)
    dump = os.path.join(save_folder, f'params.yaml')
    #with open(dump, 'w') as f:
    #    yaml.dump(args, f)

    # Define checkpointing path
    latest_path = os.path.join(load_folder_path, f'{tag}-latest.pth.tar')
    load_path = (os.path.join(load_folder_path, r_file) if r_file is not None 
        else latest_path)

    # Initialize target encoder
    target_encoder, _ = init_model(
        gt_type=gt_type,
        count_encoding=count_encoding,
        n_value_bins=n_value_bins,
        device=device,
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_special_tokens=n_special_tokens,
        n_segments=n_segments,
        n_special_values=n_special_values,
        enc_emb_dim=enc_emb_dim,
        enc_depth=enc_depth,
        pred_emb_dim=pred_emb_dim,
        pred_depth=pred_depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        use_flash_attention=use_flash_attention,
        use_layer_norm=use_layer_norm,
        api_version=api_version)

    if api_version != 'v3':
        return_layer_emb_fn = target_encoder.return_layer_emb
    else:
        return_layer_emb_fn = target_encoder.backbone.return_layer_emb

    # Initialize train and test datasets, dataloaders and samplers
    cell_dataset = make_cell_dataset(
        dataset=dataset,
        vocab_size=vocab_size,
        seq_len_cell=seq_len_cell,
        seq_len_neighborhood=seq_len_neighborhood,
        tokenizer_type=tokenizer_type,
        gt_type=gt_type,
        special_tokens=special_tokens,
        sampling_strategy=None,
        n_nonzero_tokens_list=n_nonzero_tokens)

    loader = init_dataloader_and_sampler(
        cell_dataset=cell_dataset,
        batch_size=batch_size,
        distributed=False,
        world_size=1,
        rank=0,
        collate_fn=mask_collator,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=False)
    
    _, _, target_encoder, _, _, start_epoch_ iter_number = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=None,
            predictor=None,
            target_encoder=target_encoder,
            opt=None,
            scaler=None,
            is_training=False)
    
    # Apply PEFT
    if use_peft:
        target_encoder = apply_peft(target_encoder, peft_method='lora', rank=8)

    # Convert target encoder to a classification model
    model = ClassificationModel(target_encoder, gt_type, num_classes)
    model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer (only optimize PEFT parameters)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    def save_checkpoint():

    # Run training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch}")
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for itr, (udata, _, _, masks_attention) in tqdm(enumerate(loader)):

            optimizer.zero_grad()

            tokens = udata[0].to(device, non_blocking=True)
            segments = udata[1].to(device, non_blocking=True)
            if gt_type == 'rank':
                positions = udata[2].to(device, non_blocking=True)
            elif gt_type == 'counts':
                counts = udata[2].to(device, non_blocking=True)
            masks_attention = masks_attention.to(device, non_blocking=True)

            # Get class logits
            if gt_type == 'rank':
                logits = model(
                    tokens=tokens,
                    segments=segments,
                    positions=positions,
                    masks_attention=masks_attention)
            elif gt_type == 'counts':
                logits = model(
                    tokens=tokens,
                    segments=segments,
                    counts=counts,
                    masks_attention=masks_attention)

            # Forward pass
            loss = criterion(logits, labels)  # Compute the loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        epoch_loss = running_loss / len(loader)
        accuracy = correct_preds / total_preds
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

    #log_stats()
    if LOCAL_RANK == 0:
        wandb.log(
            {"loss": loss,
            'lr':_new_lr,
            'epoch': epoch,
            'global_norm_enc': grad_stats.global_norm,
            'global_norm_pred': grad_stats_pred.global_norm,
            })
    assert not np.isnan(loss), 'loss is nan'
    if itr % checkpoint_freq_iter == 0:
        logger.info(f'Saving checkpoint at epoch {epoch} iteration {itr}')
        save_checkpoint(epoch, itr // checkpoint_freq_iter)

# Save checkpoint
logger.info('avg. loss %.3f' % loss_meter.avg)
save_checkpoint(epoch)
