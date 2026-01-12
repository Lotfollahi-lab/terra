"""
Sample Directory structure created by NestedStratifiedGroupKFold (if K_outer = 2, K_inner = 3):
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
            ├── nested_cv_visualization.png
"""

import os
import pickle
from pathlib import Path
from collections import Counter
from typing import Union, Optional, List

import numpy as np
import anndata as ad
from sklearn.model_selection import StratifiedGroupKFold

from datasets import Dataset


def filter_dataset_by_cell_ids(
        dataset: Dataset,
        target_cell_ids: List[str],
        batch_size: int = 10_000,
        num_proc: int = 4,
        temp_dir: Optional[Union[str, Path]] = None,
    ) -> Dataset:
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


class NestedStratifiedGroupKFold:
    """
    Create nested stratified group k-fold cross-validation splits.
    
    This class encapsulates the logic for creating nested stratified group k-fold
    cross-validation splits and storing configuration parameters.
    """
    
    def __init__(
            self,
            stratify_group: Union[str, List[str]] = "Patient ID",
            label_column: str = "niche19",
            K_outer: int = 4,
            K_inner: int = 3,
            shuffle: bool = True,
            seed: int = 42,
            require_all_labels: bool = False,
            max_retries: int = 10,
            group_balance_tolerance: int = 1,
            batch_size: int = 10_000,
            num_proc: int = 4,
        ):
        """
        Initialize NestedStratifiedGroupKFold with configuration parameters.
        
        Parameters
        ----------
        stratify_group : str or list[str]
            Column name(s) in adata.obs to use as groups.
        label_column : str
            Column in adata.obs containing labels for stratification.
        K_outer : int
            Number of outer folds for test evaluation.
        K_inner : int
            Number of inner folds for hyperparameter tuning.
        shuffle : bool
            Whether to shuffle before splitting.
        seed : int
            Random seed for reproducibility.
        require_all_labels : bool
            If True, ensures all labels are present in both sides of each split.
        max_retries : int
            Maximum number of random seed attempts to find valid splits.
        group_balance_tolerance : int
            Maximum allowed difference in group counts between splits.
        batch_size : int
            Batch size for dataset processing.
        num_proc : int
            Number of processes for parallel processing.
        """
        self.stratify_group = stratify_group
        self.label_column = label_column
        self.K_outer = K_outer
        self.K_inner = K_inner
        self.shuffle = shuffle
        self.seed = seed
        self.require_all_labels = require_all_labels
        self.max_retries = max_retries
        self.group_balance_tolerance = group_balance_tolerance
        self.batch_size = batch_size
        self.num_proc = num_proc

    
    def design_splits(
            self,
            adata: ad.AnnData,
        ) -> dict:
        """
        Design and validate nested stratified group k-fold cross-validation splits.
        
        This method performs Phase 1 (fast validation) without expensive I/O operations.
        It validates splits and computes indices, but does not create any files.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data matrix with obs columns for stratify_group and label_column.
        
        Returns
        -------
        split_info : dict
            Dictionary containing split metadata and validation results with indices.
            This can be passed to build_split_data() to create the actual files.
        """
        # ------------------------------------------------------------------------------------    
        assert self.K_outer >= 2, "K_outer must be at least 2"
        assert self.K_inner >= 0, "K_inner must be at least 1"
        
        # create inner split: train vs. val
        if self.K_inner == 1:
            print(f"  Number of inner folds is 1. Setting K_inner to 3 for compatibility and storing just the first split.")
            K_inner_compatible = 3
        else:
            K_inner_compatible = self.K_inner

        # ------------------------------------------------------------------------------------    
        # Get group values and labels
        if isinstance(self.stratify_group, str):
            groups = adata.obs[self.stratify_group].astype(str).values
        elif isinstance(self.stratify_group, list):
            # Combine multiple columns with "_" separator
            group_parts = [adata.obs[col].astype(str).values for col in self.stratify_group]
            # Combine columns element-wise
            groups = np.array(["_".join(parts) for parts in zip(*group_parts)])
        else:
            raise TypeError(
                f"stratify_group must be str or list[str], got {type(self.stratify_group)}"
            )
        labels = adata.obs[self.label_column].astype(str).values
        global_indices = adata.obs.index.values
        num_cells = len(adata)
        
        # Format stratify_group for display
        if isinstance(self.stratify_group, list):
            stratify_group_display = "_".join(self.stratify_group)
        else:
            stratify_group_display = self.stratify_group
        
        # Print summary
        print("=" * 60)
        print(f"Total cells: {num_cells}")
        print(f"Num. unique groups (`{stratify_group_display}`): {len(np.unique(groups))}")
        print(f"Num. unique labels (`{self.label_column}`): {len(np.unique(labels))}")
        print(f"Outer folds (test): {self.K_outer} | Inner folds (train-val): {self.K_inner}")
        print(f"Require all labels in splits: {self.require_all_labels}")
        print("=" * 60)
        
        # ------------------------------------------------------------------------------------
        # Pre-validation when require_all_labels=True
        if self.require_all_labels:
            self._pre_validate(labels, groups, K_inner_compatible)
        
        # ------------------------------------------------------------------------------------    
        # Create dictionary to store split metadata
        split_info = {
            'K_outer': self.K_outer,
            'K_inner': self.K_inner,
            'seed': self.seed,
            'stratify_group': stratify_group_display,
            'label_column': self.label_column,
            'require_all_labels': self.require_all_labels,
            'max_retries': self.max_retries if self.require_all_labels else None,
            'group_balance_tolerance': self.group_balance_tolerance if self.require_all_labels else None,
            'outer_folds': [],
            # Store validation data for build_split_data
            '_validation_data': {
                'groups': groups,
                'labels': labels,
                'global_indices': global_indices,
                'K_inner_compatible': K_inner_compatible,
            }
        }

        # ------------------------------------------------------------------------------------    
        # Phase 1: Validate all outer folds using cell indices
        print("\n" + "=" * 60)
        print("Phase 1: Validating all splits...")
        print("=" * 60)
        
        outer_fold_validations = []
        effective_max_retries = 1 if not self.require_all_labels else self.max_retries
        
        for outer_fold in range(self.K_outer):
            print(f"\n  Validating outer fold {outer_fold + 1}/{self.K_outer}...")
            result = self._validate_single_fold(
                fold_num=outer_fold,
                labels=labels,
                groups=groups,
                n_samples=num_cells,
                n_splits=self.K_outer,
                effective_max_retries=effective_max_retries,
                label_names=("Train+Val", "Test"),
            )
            
            # Store validation result (split1_pos is trainval_pos, split2_pos is test_pos)
            outer_fold_validations.append({
                'outer_fold': outer_fold,
                'effective_seed': result['effective_seed'],
                'trainval_pos': result['split1_pos'],
                'test_pos': result['split2_pos'],
                'retry_count': result['retry_count'],
            })
        
        # ------------------------------------------------------------------------------------    
        # Phase 1b: Validate all inner folds of each outer fold
        inner_fold_validations = {}  # {outer_fold: [{inner_fold, effective_seed, train_pos, val_pos, retry_count}, ...]}
        
        for outer_validation in outer_fold_validations:
            outer_fold = outer_validation['outer_fold']
            trainval_pos = outer_validation['trainval_pos']
            
            # Extract labels and groups for the train+val subset
            trainval_labels = labels[trainval_pos]
            trainval_groups = groups[trainval_pos]
            
            inner_fold_validations[outer_fold] = []
            
            for inner_fold in range(K_inner_compatible):
                
                # Only validate the first inner fold when K_inner == 1
                if self.K_inner == 1 and inner_fold > 0:
                    break
                
                print(f"      Validating inner fold {inner_fold + 1}/{K_inner_compatible} of outer fold {outer_fold + 1}...")
                result = self._validate_single_fold(
                    fold_num=inner_fold,
                    labels=trainval_labels,
                    groups=trainval_groups,
                    n_samples=len(trainval_labels),
                    n_splits=K_inner_compatible,
                    effective_max_retries=effective_max_retries,
                    label_names=("Train", "Val"),
                )
                
                # Store validation result (split1_pos is train_pos, split2_pos is val_pos)
                inner_fold_validations[outer_fold].append({
                    'inner_fold': inner_fold,
                    'effective_seed': result['effective_seed'],
                    'train_pos': result['split1_pos'],
                    'val_pos': result['split2_pos'],
                    'retry_count': result['retry_count'],
                })
        
        # Store validation results in split_info for build_split_data
        split_info['_validation_data']['outer_fold_validations'] = outer_fold_validations
        split_info['_validation_data']['inner_fold_validations'] = inner_fold_validations
        
        # ------------------------------------------------------------------------------------    
        # Print detailed summary
        print("\n" + "=" * 60)
        print("Validation Summary:")
        print("=" * 60)
        print(f"All {self.K_outer} outer folds validated successfully")
        total_inner_folds = sum(len(inner_fold_validations[of]) for of in range(self.K_outer))
        print(f"All {total_inner_folds} inner folds validated successfully")
        if self.require_all_labels:
            outer_retries = sum(1 for ov in outer_fold_validations if ov['retry_count'] > 1)
            inner_retries = sum(1 for of in range(self.K_outer) for iv in inner_fold_validations[of] if iv['retry_count'] > 1)
            if outer_retries > 0 or inner_retries > 0:
                print(f"  Outer folds requiring retries: {outer_retries}/{self.K_outer}")
                print(f"  Inner folds requiring retries: {inner_retries}/{total_inner_folds}")
            print("  All labels confirmed present in all splits")
        print()
        
        # Get unique labels for consistent ordering
        unique_labels = sorted(np.unique(labels))

        # Print detailed statistics for outer folds in table format
        print("\nOuter Folds Statistics:")
        print("-" * 100)
        print(f"{'Outer Fold':<12} | {'Label':<15} | {'Train+Val Cells':<18} | {'Train+Val Samples':<20} | {'Test Cells':<15} | {'Test Samples':<15}")
        print("-" * 100)

        for outer_validation in outer_fold_validations:
            outer_fold = outer_validation['outer_fold']
            trainval_pos = outer_validation['trainval_pos']
            test_pos = outer_validation['test_pos']
            
            trainval_labels = labels[trainval_pos]
            test_labels = labels[test_pos]
            trainval_groups = groups[trainval_pos]
            test_groups = groups[test_pos]
            
            for label in unique_labels:
                # Train+Val statistics
                trainval_label_mask = trainval_labels == label
                trainval_cells = np.sum(trainval_label_mask)
                trainval_samples = len(set(trainval_groups[trainval_label_mask]))
                
                # Test statistics
                test_label_mask = test_labels == label
                test_cells = np.sum(test_label_mask)
                test_samples = len(set(test_groups[test_label_mask]))
                
                print(f"{outer_fold + 1:<12} | {label:<15} | {trainval_cells:>16,} | {trainval_samples:>18} | {test_cells:>13,} | {test_samples:>13}")

        # Print detailed statistics for inner folds in table format
        print("\nInner Folds Statistics:")
        print("-" * 120)
        print(f"{'Outer Fold':<12} | {'Inner Fold':<12} | {'Label':<15} | {'Train Cells':<15} | {'Train Samples':<17} | {'Val Cells':<13} | {'Val Samples':<13}")
        print("-" * 120)

        for outer_validation in outer_fold_validations:
            outer_fold = outer_validation['outer_fold']
            trainval_pos = outer_validation['trainval_pos']
            
            # Get trainval labels and groups for this outer fold
            trainval_labels_local = labels[trainval_pos]
            trainval_groups_local = groups[trainval_pos]
            
            inner_folds_list = inner_fold_validations[outer_fold]
            if self.K_inner == 1:
                # When K_inner == 1, only process the first inner fold
                inner_folds_list = inner_folds_list[:1]
            
            for inner_validation in inner_folds_list:
                inner_fold = inner_validation['inner_fold']
                train_pos_inner = inner_validation['train_pos']
                val_pos_inner = inner_validation['val_pos']
                
                # Extract train and val labels/groups (relative to trainval_pos)
                train_labels = trainval_labels_local[train_pos_inner]
                val_labels = trainval_labels_local[val_pos_inner]
                train_groups = trainval_groups_local[train_pos_inner]
                val_groups = trainval_groups_local[val_pos_inner]
                
                for label in unique_labels:
                    # Train statistics
                    train_label_mask = train_labels == label
                    train_cells = np.sum(train_label_mask)
                    train_samples = len(set(train_groups[train_label_mask]))
                    
                    # Val statistics
                    val_label_mask = val_labels == label
                    val_cells = np.sum(val_label_mask)
                    val_samples = len(set(val_groups[val_label_mask]))
                    
                    print(f"{outer_fold + 1:<12} | {inner_fold + 1:<12} | {label:<15} | {train_cells:>13,} | {train_samples:>15} | {val_cells:>11,} | {val_samples:>11}")

        print("=" * 120)
        
        return split_info

    
    def build_split_data(
            self,
            adata: ad.AnnData,
            dataset: Dataset,
            split_info: dict,
            root_save_dir: Union[str, Path] = "./",
        ) -> dict:
        """
        Build split datasets and files from validated split indices.
        
        This method performs Phase 2 (expensive I/O operations) using split_info
        from design_splits(). It creates all the actual files and datasets.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data matrix with obs columns for stratify_group and label_column.
        dataset : Dataset
            Tokenized Hugging Face dataset with 'cell_id' column.
        split_info : dict
            Split metadata dictionary from design_splits() containing validation results.
        root_save_dir : str or Path
            Root directory to save the splits.
        
        Returns
        -------
        split_info : dict
            Updated dictionary containing metadata about the splits (same structure as input).
        """
        # Extract validation data
        validation_data = split_info['_validation_data']
        groups = validation_data['groups']
        global_indices = validation_data['global_indices']
        outer_fold_validations = validation_data['outer_fold_validations']
        inner_fold_validations = validation_data['inner_fold_validations']
        
        # ------------------------------------------------------------------------------------    
        # Create root save directory if it does not exist
        if isinstance(root_save_dir, str):
            root_save_dir = Path(root_save_dir)
        data_dir = root_save_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nPhase 2: Executing expensive operations (dataset filtering, file I/O)...")
        print("=" * 60)
        print(f"Root save directory: {root_save_dir}")
        print("=" * 60)
        
        # ------------------------------------------------------------------------------------
        # Prepare label string to integer index mapping
        label_cats = sorted(adata.obs[self.label_column].unique().tolist())
        label_label2idx = {lbl: i for i, lbl in enumerate(label_cats)}
        adata.obs[f'{self.label_column}_cls_idx'] = (adata.obs[self.label_column].map(label_label2idx).astype('Int64'))

        # Save the mapping to uns
        adata.uns[f'{self.label_column}_label2idx'] = label_label2idx
        adata.uns[f'{self.label_column}_idx2label'] = {str(i): lbl for lbl, i in label_label2idx.items()}
        adata.uns[f'{self.label_column}_cls_idx_num_classes'] = len(adata.obs[f'{self.label_column}_cls_idx'].unique())
        
        adata.write_h5ad(data_dir / "adata.h5ad")
        
        # ------------------------------------------------------------------------------------    
        filtered_dataset = filter_dataset_by_cell_ids(
                dataset=dataset,
                target_cell_ids=adata.obs['cell_id'].values,
                batch_size=self.batch_size,
                num_proc=self.num_proc,
                temp_dir=data_dir / "dataset_cache",
            )
        filtered_dataset.save_to_disk(
            dataset_path=str(data_dir / "dataset.dataset"),
            num_shards=1,
            num_proc=self.num_proc,
        )

        # ------------------------------------------------------------------------------------    
        # Phase 2: Execute expensive operations for all outer folds
        for outer_validation in outer_fold_validations:
            outer_fold = outer_validation['outer_fold']
            effective_seed = outer_validation['effective_seed']
            trainval_pos = outer_validation['trainval_pos']
            test_pos = outer_validation['test_pos']
            
            # ------------------------------------------------------------
            # create outer fold directory
            outer_fold_dir = data_dir / f"outer-fold_{outer_fold + 1}"
            outer_fold_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nOuter Fold {outer_fold + 1}/{self.K_outer}")
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
            if self.require_all_labels and effective_seed != self.seed:
                print(f"  Effective seed used: {effective_seed} (original: {self.seed})")
            
            # ------------------------------------------------------------
            # store outer fold info
            outer_fold_info = {
                'outer_fold': outer_fold + 1,
                'test_groups': list(test_groups),
                'trainval_groups': list(trainval_groups),
                'test_groups_count': len(test_groups),
                'trainval_groups_count': len(trainval_groups),
                'test_cells': len(test_idx),
                'trainval_cells': len(trainval_idx),
                'effective_seed': effective_seed,
                'inner_folds': []
            }

            # ------------------------------------------------------------
            # create and save test set for current outer fold
            test_adata = adata[test_idx].copy()
            test_adata.write_h5ad(outer_fold_dir / "test.h5ad")
            print(f"  Test Set for Outer Fold {outer_fold+1} saved at: {outer_fold_dir / 'test.h5ad'}")
            
            self._save_split_dataset(
                adata=test_adata,
                dataset=filtered_dataset,
                split_name=f"Test Set for Outer Fold {outer_fold+1}",
                save_dir=outer_fold_dir,
                filename="test.dataset",
            )
            
            # ------------------------------------------------------------
            trainval_adata = adata[trainval_idx].copy()
            trainval_adata.write_h5ad(outer_fold_dir / "train.h5ad")
            print(f"  Train Set for Outer Fold {outer_fold+1} saved at: {outer_fold_dir / 'train.h5ad'}")
            
            self._save_split_dataset(
                adata=trainval_adata,
                dataset=filtered_dataset,
                split_name=f"Train Set for Outer Fold {outer_fold+1}",
                save_dir=outer_fold_dir,
                filename="train.dataset",
            )

            # ------------------------------------------------------------
            # Phase 2b: Execute expensive operations for all inner folds of this outer fold
            # Use stored validation results
            inner_folds_to_process = inner_fold_validations[outer_fold]
            if self.K_inner == 1:
                # When K_inner == 1, only process the first inner fold
                inner_folds_to_process = inner_folds_to_process[:1]
            
            for inner_validation in inner_folds_to_process:
                inner_fold = inner_validation['inner_fold']
                effective_seed_inner = inner_validation['effective_seed']
                train_pos_inner = inner_validation['train_pos']
                val_pos_inner = inner_validation['val_pos']
                
                # Note: train_pos_inner and val_pos_inner are relative to trainval_pos
                # We need to map them to the original indices
                # Get trainval_groups for this outer fold
                trainval_groups_local = groups[trainval_pos]
                
                # ---------------------------------------------
                # create inner fold directory
                inner_fold_dir = outer_fold_dir / f"inner-fold_{inner_fold + 1}"
                inner_fold_dir.mkdir(parents=True, exist_ok=True)

                print(f"\nInner Fold {inner_fold + 1}/{self.K_inner}")
                print("-" * 40)
                print(f"Directory created at: {inner_fold_dir}")

                # ---------------------------------------------
                # map inner fold positions back to original adata indices
                val_idx = trainval_idx[val_pos_inner]
                train_idx = trainval_idx[train_pos_inner]

                # ---------------------------------------------
                # print and store inner fold statistics
                val_groups = sorted(set(trainval_groups_local[val_pos_inner]))
                train_groups = sorted(set(trainval_groups_local[train_pos_inner]))
                
                print(f"  Val groups ({len(val_groups)}): {val_groups}")
                print(f"  Train groups ({len(train_groups)}): {train_groups}")
                print(f"  Num. Val cells: {len(val_idx):,}")
                print(f"  Num. Train cells: {len(train_idx):,}")
                if self.require_all_labels and effective_seed_inner != self.seed:
                    print(f"  Effective seed used: {effective_seed_inner} (original: {self.seed})")

                # ---------------------------------------------
                # store inner fold info
                inner_fold_info = {
                    'inner_fold': inner_fold + 1,
                    'val_groups': list(val_groups),
                    'train_groups': list(train_groups),
                    'val_groups_count': len(val_groups),
                    'train_groups_count': len(train_groups),
                    'val_cells': len(val_idx),
                    'train_cells': len(train_idx),
                    'effective_seed': effective_seed_inner,
                }
                outer_fold_info['inner_folds'].append(inner_fold_info)
                
                # ---------------------------------------------
                # create val set for current inner fold
                val_adata = adata[val_idx].copy()
                val_adata.write_h5ad(inner_fold_dir / "val.h5ad")
                print(f"  Val Set for Inner Fold {inner_fold+1} saved at: {inner_fold_dir / 'val.h5ad'}")
                
                self._save_split_dataset(
                    adata=val_adata,
                    dataset=filtered_dataset,
                    split_name=f"Val Set for Inner Fold {inner_fold+1}",
                    save_dir=inner_fold_dir,
                    filename="val.dataset",
                )

                # ---------------------------------------------
                # create train set for current inner fold
                train_adata = adata[train_idx].copy()
                train_adata.write_h5ad(inner_fold_dir / "train.h5ad")
                print(f"  Train Set for Inner Fold {inner_fold+1} saved at: {inner_fold_dir / 'train.h5ad'}")
                
                self._save_split_dataset(
                    adata=train_adata,
                    dataset=filtered_dataset,
                    split_name=f"Train Set for Inner Fold {inner_fold+1}",
                    save_dir=inner_fold_dir,
                    filename="train.dataset",
                )
                
            # ------------------------------------------------------------
            split_info['outer_folds'].append(outer_fold_info)
            print(f"\n  Outer fold {outer_fold + 1} complete.")
        
        # Remove internal validation data before saving
        split_info_clean = {k: v for k, v in split_info.items() if k != '_validation_data'}
        
        # ------------------------------------------------------------------------------------    
        # Save split metadata
        metadata_path = root_save_dir / "split_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(split_info_clean, f)
        print(f"\n{'=' * 60}")
        print(f"All splits saved to: {data_dir}")
        print(f"Metadata saved to: {metadata_path}")
        
        return split_info_clean

    
    def check_split_validity(
            self,
            labels_split1: np.ndarray,
            labels_split2: np.ndarray,
            all_labels: np.ndarray,
        ) -> bool:
        """
        Check if all labels are present in both splits (returns boolean, doesn't raise).
        
        Parameters
        ----------
        labels_split1 : array-like
            Labels from the first split.
        labels_split2 : array-like
            Labels from the second split.
        all_labels : array-like
            All labels that should be present in both splits.
        
        Returns
        -------
        bool
            True if all labels are present in both splits, False otherwise.
        """
        unique_labels_split1 = set(np.unique(labels_split1))
        unique_labels_split2 = set(np.unique(labels_split2))
        all_labels_set = set(np.unique(all_labels))
        
        missing_in_split1 = all_labels_set - unique_labels_split1
        missing_in_split2 = all_labels_set - unique_labels_split2
        
        return len(missing_in_split1) == 0 and len(missing_in_split2) == 0

    
    def check_group_balance_per_label(
            self,
            labels: np.ndarray,
            groups_split1: np.ndarray,
            labels_split1: np.ndarray,
            groups_split2: np.ndarray,
            labels_split2: np.ndarray,
            tolerance: int = None,
            label_names: tuple[str, str] = ("Split1", "Split2"),
        ) -> tuple[bool, str]:
        """
        Check if the number of unique groups is approximately equal between splits for each label.
        
        Parameters
        ----------
        labels : np.ndarray
            All labels in the dataset.
        groups_split1 : np.ndarray
            Groups from the first split.
        labels_split1 : np.ndarray
            Labels from the first split.
        groups_split2 : np.ndarray
            Groups from the second split.
        labels_split2 : np.ndarray
            Labels from the second split.
        tolerance : int, optional
            Maximum allowed difference in group counts per label. If None, uses self.group_balance_tolerance.
        label_names : tuple[str, str]
            Names for the two splits (e.g., ("Test", "Train+Val") or ("Train", "Val")).
        
        Returns
        -------
        tuple[bool, str]
            (True, "") if balanced for all labels, (False, error_message) otherwise.
        """
        if tolerance is None:
            tolerance = self.group_balance_tolerance
        
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            # Get groups for this label in each split
            mask_split1 = labels_split1 == label
            mask_split2 = labels_split2 == label
            
            groups_label_split1 = groups_split1[mask_split1]
            groups_label_split2 = groups_split2[mask_split2]
            
            unique_groups_split1 = len(set(groups_label_split1))
            unique_groups_split2 = len(set(groups_label_split2))
            difference = abs(unique_groups_split1 - unique_groups_split2)
            
            if difference > tolerance:
                return False, f"Label '{label}': {label_names[0]}={unique_groups_split1} groups, {label_names[1]}={unique_groups_split2} groups, difference={difference} > {tolerance}"
        
        return True, ""


    def _print_split_statistics(
            self,
            split1_labels: np.ndarray,
            split2_labels: np.ndarray,
            all_labels: np.ndarray,
            split1_groups: np.ndarray,
            split2_groups: np.ndarray,
            label_names: tuple[str, str],
        ):
        """
        Print label statistics and cell counts per label for a split.
        
        Parameters
        ----------
        split1_labels : np.ndarray
            Labels from the first split.
        split2_labels : np.ndarray
            Labels from the second split.
        all_labels : np.ndarray
            All labels in the dataset.
        split1_groups : np.ndarray
            Groups from the first split.
        split2_groups : np.ndarray
            Groups from the second split.
        label_names : tuple[str, str]
            Names for the two splits (e.g., ("Test", "Train+Val") or ("Train", "Val")).
        """
        # Print label statistics
        split1_labels_unique = len(np.unique(split1_labels))
        split2_labels_unique = len(np.unique(split2_labels))
        total_labels_unique = len(np.unique(all_labels))
        print(f"          Label statistics: {label_names[0]}={split1_labels_unique}, {label_names[1]}={split2_labels_unique}, Total={total_labels_unique}")
        # Print cell counts per label
        split1_label_counts = Counter(split1_labels)
        split2_label_counts = Counter(split2_labels)
        all_labels_sorted = sorted(np.unique(all_labels))
        print(f"          Cell counts per label:")
        for label in all_labels_sorted:
            split1_count = split1_label_counts.get(label, 0)
            split2_count = split2_label_counts.get(label, 0)
            # Count unique groups/samples with this label
            split1_label_mask = split1_labels == label
            split2_label_mask = split2_labels == label
            split1_groups_with_label = len(set(split1_groups[split1_label_mask]))
            split2_groups_with_label = len(set(split2_groups[split2_label_mask]))
            print(f"            {label}: {label_names[0]}={split1_count} cells ({split1_groups_with_label} samples), {label_names[1]}={split2_count} cells ({split2_groups_with_label} samples)")


    def _save_split_dataset(
            self,
            adata: ad.AnnData,
            dataset: Dataset,
            split_name: str,
            save_dir: Path,
            filename: str,
        ):
        """
        Filter dataset by cell IDs from adata and save to disk.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data matrix with 'cell_id' column.
        dataset : Dataset
            Hugging Face dataset to filter.
        split_name : str
            Name of the split (e.g., "Test Set for Outer Fold 1").
        save_dir : Path
            Directory where to save the dataset.
        filename : str
            Filename for the dataset (e.g., "test.dataset").
        """
        cache_dir_name = filename.replace('.dataset', '_dataset_cache')
        filtered_dataset = filter_dataset_by_cell_ids(
            dataset=dataset,
            target_cell_ids=adata.obs['cell_id'].values,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            temp_dir=save_dir / cache_dir_name,
        )
        dataset_path = save_dir / filename
        filtered_dataset.save_to_disk(
            dataset_path=str(dataset_path),
            num_shards=1,
            num_proc=self.num_proc,
        )
        print(f"  {split_name} saved at: {dataset_path}")
      
  
    def _validate_single_fold(
            self,
            fold_num: int,
            labels: np.ndarray,
            groups: np.ndarray,
            n_samples: int,
            n_splits: int,
            effective_max_retries: int,
            label_names: tuple[str, str] = ("Split1", "Split2"),
        ) -> dict:
        """
        Validate a single fold by trying different seeds until a valid split is found.
        
        Parameters
        ----------
        fold_num : int
            The fold number (0-indexed).
        labels : np.ndarray
            Labels for stratification.
        groups : np.ndarray
            Groups for grouping.
        n_samples : int
            Number of samples.
        n_splits : int
            Number of splits for StratifiedGroupKFold.
        effective_max_retries : int
            Maximum number of retry attempts.
        label_names : tuple[str, str]
            Names for the two splits (e.g., ("Test", "Train+Val") or ("Train", "Val")).
        
        Returns
        -------
        dict
            Dictionary with keys: 'effective_seed', 'split1_pos', 'split2_pos', 'retry_count'
        """
        valid_split_found = False
        effective_seed = self.seed
        split1_pos = None
        split2_pos = None
        retry_count = 0
        
        for retry in range(effective_max_retries):
            try_seed = self.seed + retry
            print(f"         Attempt {retry + 1}/{effective_max_retries} with seed={try_seed}...", end=" ")
            cv = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=self.shuffle,
                random_state=try_seed
            )
            # Get the split for this specific fold
            splits = list(cv.split(X=np.zeros(n_samples), y=labels, groups=groups))
            split1_pos_candidate, split2_pos_candidate = splits[fold_num]
            
            # Extract labels and groups once
            split1_labels_candidate = labels[split1_pos_candidate]
            split2_labels_candidate = labels[split2_pos_candidate]
            split1_groups_candidate = groups[split1_pos_candidate]
            split2_groups_candidate = groups[split2_pos_candidate]
            
            # Check label validity if required
            if self.require_all_labels:
                is_valid = self.check_split_validity(
                    labels_split1=split1_labels_candidate,
                    labels_split2=split2_labels_candidate,
                    all_labels=labels,
                )
                
                if not is_valid:
                    print("Valid split not found (missing labels in splits)")
                    continue
            
            # Check group balance per label (always done)
            is_balanced, balance_error_msg = self.check_group_balance_per_label(
                labels=labels,
                groups_split1=split1_groups_candidate,
                labels_split1=split1_labels_candidate,
                groups_split2=split2_groups_candidate,
                labels_split2=split2_labels_candidate,
                label_names=label_names,
            )
            
            if not is_balanced:
                print(f"Invalid split (group imbalance: {balance_error_msg})")
                continue
            
            # Valid split found
            if self.require_all_labels:
                print("Valid split found (all labels present in both splits)")
            else:
                print("Valid split found (group balance validated)")
            
            # Print label statistics using helper function
            self._print_split_statistics(
                split1_labels=split1_labels_candidate,
                split2_labels=split2_labels_candidate,
                all_labels=labels,
                split1_groups=split1_groups_candidate,
                split2_groups=split2_groups_candidate,
                label_names=label_names,
            )
            
            valid_split_found = True
            effective_seed = try_seed
            split1_pos = split1_pos_candidate
            split2_pos = split2_pos_candidate
            retry_count = retry + 1
            break
        
        if not valid_split_found:
            error_details = (
                "All labels must be present in both splits. " if self.require_all_labels else ""
            )
            raise ValueError(
                f"Could not find valid split for fold {fold_num + 1} after {effective_max_retries} attempts. "
                f"{error_details}"
                f"Try increasing max_retries or check if your data supports this requirement."
            )
        
        return {
            'effective_seed': effective_seed,
            'split1_pos': split1_pos,
            'split2_pos': split2_pos,
            'retry_count': retry_count,
        }

    
    def _pre_validate(
            self,
            labels: np.ndarray,
            groups: np.ndarray,
            K_inner_compatible: int,
        ):
        """
        Called when require_all_labels=True. Checks that each label appears in at least K_outer groups for outer folds and at least K_inner_compatible groups for inner folds.
        
        Parameters
        ----------
        labels : np.ndarray
            All labels in the dataset.
        groups : np.ndarray
            All groups in the dataset.
        K_inner_compatible : int
            Compatible number of inner folds.
        """
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            label_mask = labels == label
            groups_with_label = set(groups[label_mask])
            n_groups_with_label = len(groups_with_label)
            
            # Check that each label appears in at least K_outer groups for outer folds
            if n_groups_with_label < self.K_outer:
                raise ValueError(
                    f"Label '{label}' appears in only {n_groups_with_label} group(s), "
                    f"but {self.K_outer} groups are required for require_all_labels=True. "
                    f"Each label must appear in at least K_outer groups to be split across outer folds."
                )
        
            # Check that each label appears in enough groups for inner folds
            if n_groups_with_label < K_inner_compatible:
                raise ValueError(
                    f"Label '{label}' appears in only {n_groups_with_label} group(s), "
                    f"but {K_inner_compatible} groups are required for require_all_labels=True. "
                    f"This ensures that after outer fold splitting, at least {K_inner_compatible} groups "
                    f"remain in Train+Val for inner fold splitting. "
                    f"Each label must appear in at least (K_outer-1 + K_inner) groups total."
                )