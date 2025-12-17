import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def filter_dataset_by_cell_ids(
        dataset,
        target_cell_ids,
        batch_size=10000,
        num_proc=4,
        temp_dir=None,
    ):
    """
    Filter a Hugging Face dataset to keep only cells matching a given list of cell IDs.
    
    Uses a directory to avoid permission issues when filtering large datasets.

    Parameters:
    -----------
    dataset : Dataset
        Hugging Face dataset with 'cell_id' column
    target_cell_ids : list
        List of complete cell IDs to filter for (e.g., ['1000_batch1_0', '1000_batch1_1', '1002_batch3_5'])
    batch_size : int, default=10000
        Batch size for processing
    num_proc : int, default=4
        Number of processes for parallel processing
    temp_dir : str, optional
        Custom temporary directory path. If None, uses current working directory.
        
    Returns:
    --------
    Dataset
        Filtered dataset containing only matching items.
    """    
    # Convert to set for faster lookup
    target_cell_ids_set = set(target_cell_ids)
    
    # Define the filter function
    def filter_function(ids):
        return [cell_id in target_cell_ids_set for cell_id in ids]
    
    # Cleanup function
    def cleanup_cache_file(cache_file_path):
        """Clean up cache file."""
        try:
            if os.path.exists(cache_file_path):
                os.remove(cache_file_path)
            # Also clean up the cache directory if it's empty
            cache_dir = os.path.dirname(cache_file_path)
            if os.path.exists(cache_dir) and not os.listdir(cache_dir):
                os.rmdir(cache_dir)
        except Exception:
            pass  # Silently fail if cleanup doesn't work

    # Determine temp directory
    if temp_dir is None:
        temp_dir = os.getcwd()
    
    # Create directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    cache_file = os.path.join(temp_dir, "filtered_dataset_cache_by_cell_ids.arrow")
    
    try:
        # Apply the filter with cache location
        filtered_dataset = dataset.filter(
            filter_function,
            input_columns="cell_id",
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            cache_file_name=cache_file,
        )
        
        # Clean up cache file after successful filtering
        cleanup_cache_file(cache_file)
        
        return filtered_dataset
    except Exception as e:
        # Clean up on error
        cleanup_cache_file(cache_file)
        raise
    

def plot_nested_cv_from_split_info(
        split_info: dict,
        adata,
        save_path=None,
        ax=None,
        figsize=(14, 10),
        cmap_data=None,
        title: str = "Nested StratifiedGroupKFold",
    ):
    """
    Visualize nested stratified group k-fold cross-validation from split_info.
    
    Parameters
    ----------
    split_info : dict
        Split metadata dictionary from nested_stratified_group_k_fold_splits().
    adata : AnnData
        Annotated data matrix with obs columns for stratify_group and label_column.
    save_path : str or Path, optional
        Path to save the figure. If None, figure is not saved.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    figsize : tuple
        Figure size if creating new figure.
    cmap_data : colormap, optional
        Colormap for class/group visualization.
    title : str
        Plot title.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes with the visualization.
    """
    # Extract metadata from split_info
    stratify_group = split_info['stratify_group']
    label_column = split_info['label_column']
    
    # Extract data from adata
    groups = adata.obs[stratify_group].astype(str).values
    labels = adata.obs[label_column].astype(str).values
    n_samples = len(adata)
    
    # Convert to numeric for coloring
    unique_groups = np.unique(groups)
    unique_labels = np.unique(labels)
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    
    groups_numeric = np.array([group_to_idx[g] for g in groups])
    labels_numeric = np.array([label_to_idx[l] for l in labels])
    
    # Set up colormap for data
    if cmap_data is None:
        cmap_data = plt.cm.tab20
    
    # Define colors
    color_train = [0.27, 0.51, 0.71, 1.0]      # Blue
    color_val_test = [0.99, 0.55, 0.38, 1.0]   # Orange
    color_heldout = [0.8, 0.8, 0.8, 1.0]       # Gray
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Track current row and labels
    current_row = 0
    y_labels = []
    y_positions = []
    
    # Helper function to get sample indices for a set of groups
    def get_indices_for_groups(group_list):
        group_set = set(group_list)
        return np.array([i for i, g in enumerate(groups) if g in group_set])
    
    # Iterate through outer folds from split_info
    for outer_fold_info in split_info['outer_folds']:
        outer_fold = outer_fold_info['outer_fold']
        test_groups = set(outer_fold_info['test_groups'])
        trainval_groups = set(outer_fold_info['trainval_groups'])
        
        # Plot outer fold (train+val vs test)
        colors = np.zeros((n_samples, 4))
        for i in range(n_samples):
            if groups[i] in test_groups:
                colors[i] = color_val_test  # Test = orange
            else:
                colors[i] = color_train     # Train+Val = blue
        
        ax.scatter(
            range(n_samples),
            [current_row + 0.5] * n_samples,
            c=colors,
            marker="_",
            lw=10,
        )
        y_labels.append(f"Outer {outer_fold}")
        y_positions.append(current_row + 0.5)
        current_row += 1
        
        # Plot inner folds
        for inner_fold_info in outer_fold_info['inner_folds']:
            inner_fold = inner_fold_info['inner_fold']
            val_groups = set(inner_fold_info['val_groups'])
            train_groups = set(inner_fold_info['train_groups'])
            
            # Color each sample
            colors = np.zeros((n_samples, 4))
            for i in range(n_samples):
                if groups[i] in test_groups:
                    colors[i] = color_heldout    # Held-out test = gray
                elif groups[i] in val_groups:
                    colors[i] = color_val_test   # Val = orange
                elif groups[i] in train_groups:
                    colors[i] = color_train      # Train = blue
                else:
                    colors[i] = color_heldout    # Shouldn't happen
            
            ax.scatter(
                range(n_samples),
                [current_row + 0.5] * n_samples,
                c=colors,
                marker="_",
                lw=8,
            )
            y_labels.append(f"  Inner {inner_fold}")
            y_positions.append(current_row + 0.5)
            current_row += 1
    
    # Add spacing before class/group rows
    current_row += 0.5
    
    # Plot class row
    ax.scatter(
        range(n_samples),
        [current_row + 0.5] * n_samples,
        c=labels_numeric,
        marker="_",
        lw=10,
        cmap=cmap_data,
    )
    y_labels.append("class")
    y_positions.append(current_row + 0.5)
    current_row += 1
    
    # Plot group row
    ax.scatter(
        range(n_samples),
        [current_row + 0.5] * n_samples,
        c=groups_numeric,
        marker="_",
        lw=10,
        cmap=cmap_data,
    )
    y_labels.append("group")
    y_positions.append(current_row + 0.5)
    
    # Configure axes
    ax.set(
        yticks=y_positions,
        yticklabels=y_labels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[current_row + 1.5, -0.5],
        xlim=[-0.5, n_samples - 0.5],
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # Add legend
    legend_elements = [
        Patch(facecolor=color_val_test, label="Validation/Test set"),
        Patch(facecolor=color_train, label="Training set"),
        Patch(facecolor=color_heldout, label="Held-out test (outer)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.22, 1.0),
    )
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"CV visualization saved to: {save_path}")
    
    return fig, ax


def nested_stratified_group_k_fold_splits(
        adata,
        dataset,
        stratify_group: str = "Patient ID",
        label_column: str = "niche19",
        K_outer: int = 4,
        K_inner: int = 3,
        shuffle: bool = True,
        seed: int = 42,
        root_save_dir: str | Path = "./",
    ):
    """
    Create nested stratified group k-fold cross-validation splits.
    
    Sample Directory structure if K_outer = 2, K_inner = 3:
        root_save_dir/
            ├── data/
                ├── outer-fold_1/
                    ├── test.h5ad
                    ├── test.dataset
                    ├── train.h5ad
                    ├── train.dataset
                    ├── inner-fold_1/
                        ├── train.h5ad
                        ├── train.dataset
                        ├── val.h5ad
                        ├── val.dataset
                    ├── inner-fold_2/
                        ├── train.h5ad
                        ├── train.dataset
                        ├── val.h5ad
                        ├── val.dataset
                    ├── inner-fold_3/
                        ├── train.h5ad
                        ├── train.dataset
                        ├── val.h5ad
                        ├── val.dataset
            ├── outer-fold_2/
                    ├── test.h5ad
                    ├── test.dataset
                    ├── train.h5ad
                    ├── train.dataset
                    ├── inner-fold_1/
                        ├── train.h5ad
                        ├── train.dataset
                        ├── val.h5ad
                        ├── val.dataset
                    ├── inner-fold_2/
                        ├── train.h5ad
                        ├── train.dataset
                        ├── val.h5ad
                        ├── val.dataset
                    ├── inner-fold_3/
                        ├── train.h5ad
                        ├── train.dataset
                        ├── val.h5ad
                        ├── val.dataset
            ├── split_metadata.pkl
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with obs columns for stratify_group and label_column.
    dataset : Dataset
        Tokenized Hugging Face dataset with 'cell_id' column.
    stratify_group : str
        Column in adata.obs to use as groups (e.g., "Patient ID" or "tissue_section").
        All cells from the same group stay together in splits.
    label_column : str
        Column in adata.obs containing labels for stratification (e.g., "niche19").
    K_outer : int
        Number of outer folds for test evaluation.
    K_inner : int
        Number of inner folds for hyperparameter tuning within each outer fold.
    shuffle : bool
        Whether to shuffle before splitting.
    seed : int
        Random seed for reproducibility.
    root_save_dir : str or Path
        Root directory to save the splits.
    
    Returns
    -------
    split_info : dict
        Dictionary containing metadata about the splits.
    """
    # ------------------------------------------------------------------------------------    
    assert K_outer >= 2, "K_outer must be at least 2"
    assert K_inner >= 0, "K_inner must be at least 1"
    
    # create inner split: train vs. val
    if K_inner == 1:
        print(f"  Number of inner folds is 1. Setting K_inner to 3 for compatibility and storing just the first split.")
        K_inner_compatible = 3
    else:
        K_inner_compatible = K_inner

    # ------------------------------------------------------------------------------------    
    # create root_save_dir if it does not exist
    if isinstance(root_save_dir, str):
        root_save_dir = Path(root_save_dir)
    data_dir = root_save_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------------------    
    # print split information
    groups = adata.obs[stratify_group].astype(str).values
    labels = adata.obs[label_column].astype(str).values
    global_indices = adata.obs.index.values
    num_cells = len(adata)
    
    print(f"Total Cells: {num_cells}")
    print(f"Num. unique groups (`{stratify_group}`): {len(np.unique(groups))}")
    print(f"Num. unique labels (`{label_column}`): {len(np.unique(labels))}")
    print(f"Outer folds (test): {K_outer} | Inner folds (train-val): {K_inner}")
    print(f"Root save directory: {root_save_dir}")
    print("=" * 60)
    
    # ------------------------------------------------------------------------------------
    # prepare label string to integer index mapping
    label_cats = sorted(adata.obs[label_column].unique().tolist())
    label_label2idx = {lbl: i for i, lbl in enumerate(label_cats)}
    adata.obs[f'{label_column}_cls_idx'] = (adata.obs[label_column].map(label_label2idx).astype('Int64'))

    # Save the mapping to uns
    adata.uns[f'{label_column}_label2idx'] = label_label2idx
    adata.uns[f'{label_column}_idx2label'] = {str(i): lbl for lbl, i in label_label2idx.items()}
    adata.uns[f'{label_column}_cls_idx_num_classes'] = len(adata.obs[f'{label_column}_cls_idx'].unique())
    
    adata.write_h5ad(data_dir / "adata.h5ad")
    
    # ------------------------------------------------------------------------------------    
    filtered_dataset = filter_dataset_by_cell_ids(
            dataset=dataset,
            target_cell_ids=adata.obs['cell_id'].values,
            batch_size=10000,
            num_proc=4,
            temp_dir=data_dir / "dataset_cache",
        )
    filtered_dataset.save_to_disk(
        dataset_path=str(data_dir / "dataset.dataset"),
        num_shards=1,
        num_proc=4,
    )

    # ------------------------------------------------------------------------------------    
    # create dictionary to store split metadata
    split_info = {
        'K_outer': K_outer,
        'K_inner': K_inner,
        'seed': seed,
        'stratify_group': stratify_group,
        'label_column': label_column,
        'outer_folds': []
    }
    # ------------------------------------------------------------------------------------    
    # create outer split: (train,val) vs test
    outer_cv = StratifiedGroupKFold(
        n_splits=K_outer,
        shuffle=shuffle,
        random_state=seed
    )
    
    # ------------------------------------------------------------------------------------    
    # ------------------------------------------------------------------------------------    
    # iterate over outer folds
    for outer_fold, (trainval_pos, test_pos) in enumerate(
        outer_cv.split(
            X=np.zeros(num_cells),
            y=labels,
            groups=groups
        )
    ):
        # ------------------------------------------------------------
        # create outer fold directory
        outer_fold_dir = data_dir / f"outer-fold_{outer_fold + 1}"
        outer_fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nOuter Fold {outer_fold + 1}/{K_outer}")
        print("-" * 40)
        print(f"Directory created at: {outer_fold_dir}")

        # ------------------------------------------------------------
        # map outer fold positions back to original adata indices
        test_idx = global_indices[test_pos]
        trainval_idx = global_indices[trainval_pos]

        # ------------------------------------------------------------
        # print and store outer fold statistics
        test_groups = sorted(set(groups[test_pos]))
        trainval_groups = sorted(set(groups[trainval_pos]))
        
        print(f"  Test groups ({len(test_groups)}): {test_groups}")
        print(f"  Train+Val groups ({len(trainval_groups)}): {trainval_groups}")
        print(f"  Num. Test cells: {len(test_idx):,}")
        print(f"  Num. Train+Val cells: {len(trainval_idx):,}")
        
        # ------------------------------------------------------------
        # store outer fold info
        outer_fold_info = {
            'outer_fold': outer_fold + 1,
            'test_groups': list(test_groups),
            'trainval_groups': list(trainval_groups),
            'test_cells': len(test_idx),
            'trainval_cells': len(trainval_idx),
            'inner_folds': []
        }

        # ------------------------------------------------------------
        # create and save test set for current outer fold
        test_adata = adata[test_idx].copy()
        test_adata.write_h5ad(outer_fold_dir / "test.h5ad")
        print(f"  Test Set for Outer Fold {outer_fold+1} saved at: {outer_fold_dir / 'test.h5ad'}")
        
        test_dataset = filter_dataset_by_cell_ids(
            dataset=filtered_dataset,
            target_cell_ids=test_adata.obs['cell_id'].values,
            batch_size=10000,
            num_proc=4,
            temp_dir=outer_fold_dir / "test_dataset_cache",
        )
        test_dataset.save_to_disk(
            dataset_path=str(outer_fold_dir / "test.dataset"),
            num_shards=1,
            num_proc=4,
        )
        print(f"  Test Set for Outer Fold {outer_fold+1} saved at: {outer_fold_dir / 'test.dataset'}")
        
        # ------------------------------------------------------------
        trainval_adata = adata[trainval_idx].copy()
        trainval_adata.write_h5ad(outer_fold_dir / "train.h5ad")
        print(f"  Train Set for Outer Fold {outer_fold+1} saved at: {outer_fold_dir / 'train.h5ad'}")
        
        trainval_dataset = filter_dataset_by_cell_ids(
            dataset=filtered_dataset,
            target_cell_ids=trainval_adata.obs['cell_id'].values,
            batch_size=10000,
            num_proc=4,
            temp_dir=outer_fold_dir / "train_dataset_cache",
        )
        trainval_dataset.save_to_disk(
            dataset_path=str(outer_fold_dir / "train.dataset"),
            num_shards=1,
            num_proc=4,
        )
        print(f"  Train Set for Outer Fold {outer_fold+1} saved at: {outer_fold_dir / 'train.dataset'}")

        # ------------------------------------------------------------
        # extract labels and groups for the train+val subset
        trainval_labels = labels[trainval_pos]
        trainval_groups = groups[trainval_pos]
        
        inner_cv = StratifiedGroupKFold(
            n_splits=K_inner_compatible,
            shuffle=shuffle,
            random_state=seed
        )
        
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # iterate over inner folds
        for inner_fold, (train_pos_inner, val_pos_inner) in enumerate(
            inner_cv.split(
                X=np.zeros(len(trainval_labels)),
                y=trainval_labels,
                groups=trainval_groups
            )
        ):
            # ---------------------------------------------
            # create inner fold directory
            inner_fold_dir = outer_fold_dir / f"inner-fold_{inner_fold + 1}"
            inner_fold_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nInner Fold {inner_fold + 1}/{K_inner}")
            print("-" * 40)
            print(f"Directory created at: {inner_fold_dir}")

            # ---------------------------------------------
            # map inner fold positions back to original adata indices
            val_idx = trainval_idx[val_pos_inner]
            train_idx = trainval_idx[train_pos_inner]

            # ---------------------------------------------
            # print and store inner fold statistics
            val_groups = sorted(set(trainval_groups[val_pos_inner]))
            train_groups = sorted(set(trainval_groups[train_pos_inner]))
            
            print(f"  Val groups ({len(val_groups)}): {val_groups}")
            print(f"  Train groups ({len(train_groups)}): {train_groups}")
            print(f"  Num. Val cells: {len(val_idx):,}")
            print(f"  Num. Train cells: {len(train_idx):,}")

            # ---------------------------------------------
            # store inner fold info
            inner_fold_info = {
                'inner_fold': inner_fold + 1,
                'val_groups': list(val_groups),
                'train_groups': list(train_groups),
                'val_cells': len(val_idx),
                'train_cells': len(train_idx),
            }
            outer_fold_info['inner_folds'].append(inner_fold_info)
            
            # ---------------------------------------------
            # create val set for current inner fold
            val_adata = adata[val_idx].copy()
            val_adata.write_h5ad(inner_fold_dir / f"val.h5ad")
            print(f"  Val Set for Inner Fold {inner_fold+1} saved at: {inner_fold_dir / 'val.h5ad'}")
            
            val_dataset = filter_dataset_by_cell_ids(
                dataset=filtered_dataset,
                target_cell_ids=val_adata.obs['cell_id'].values,
                batch_size=10000,
                num_proc=4,
                temp_dir=inner_fold_dir / "val_dataset_cache",
            )
            val_dataset.save_to_disk(
                dataset_path=str(inner_fold_dir / "val.dataset"),
                num_shards=1,
                num_proc=4,
            )
            print(f"  Val Set for Inner Fold {inner_fold+1} saved at: {inner_fold_dir / 'val.dataset'}")

            # ---------------------------------------------
            # create train set for current inner fold
            train_adata = adata[train_idx].copy()            
            train_adata.write_h5ad(inner_fold_dir / f"train.h5ad")
            print(f"  Train Set for Inner Fold {inner_fold+1} saved at: {inner_fold_dir / 'train.h5ad'}")
            
            train_dataset = filter_dataset_by_cell_ids(
                dataset=filtered_dataset,
                target_cell_ids=train_adata.obs['cell_id'].values,
                batch_size=10000,
                num_proc=4,
                temp_dir=inner_fold_dir / "train_dataset_cache",
            )
            train_dataset.save_to_disk(
                dataset_path=str(inner_fold_dir / "train.dataset"),
                num_shards=1,
                num_proc=4,
            )
            print(f"  Train Set for Inner Fold {inner_fold+1} saved at: {inner_fold_dir / 'train.dataset'}")
            
            if K_inner == 1:
                break
            
        # ------------------------------------------------------------
        split_info['outer_folds'].append(outer_fold_info)
        print(f"\n  Outer fold {outer_fold + 1} complete.")
    
    # ------------------------------------------------------------------------------------    
    # save split metadata
    metadata_path = root_save_dir / "split_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(split_info, f)
    print(f"\n{'=' * 60}")
    print(f"All splits saved to: {data_dir}")
    print(f"Metadata saved to: {metadata_path}")
    
    # ------------------------------------------------------------------------------------    
    # create and save visualization
    plot_path = root_save_dir / "nested_cv_visualization.png"
    fig, ax = plot_nested_cv_from_split_info(
        split_info=split_info,
        adata=adata,
        save_path=plot_path,
        title=f"Nested StratifiedGroupKFold (K_outer={K_outer}, K_inner={K_inner})",
    )
    plt.close(fig)  # Close figure to free memory
    
    return split_info