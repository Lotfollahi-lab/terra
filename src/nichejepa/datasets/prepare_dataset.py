import random
from typing import Tuple, Union

import datasets
from datasets import load_from_disk
from sklearn.model_selection import train_test_split


def prepare_dataset(args: dict,
                    split_dataset: bool=True
                    ) -> Union[Tuple[datasets.arrow_dataset.Dataset,
                                     datasets.arrow_dataset.Dataset],
                               datasets.arrow_dataset.Dataset]:
    """
    Prepare the dataset by loading it, determining sample size, and splitting it
    into training and test sets based on the provided configuration parameters.

    Parameters
    -----------
    args:
        A dictionary containing the configuration parameters, including:
            - data_path: The path to the dataset.
            - sample_size: The size of the dataset to sample.
            - sample_subset: Whether to sample a subset of the dataset.
            - split: The train-test split ratio.
            - stratify: Whether to stratify the dataset during the split.
            - random_state: The random seed for reproducibility.
    split_dataset:
        If 'True', split the Huggingface dataset into train and test datasets.

    Returns
    -----------
    1)
    train_dataset:
        The training portion of the dataset.
    test_dataset:
        The test portion of the dataset.
    
    2)
    dataset:
        The combined training and test portion of the dataset with a 'split'
        label.
    """
    # Load dataset from the specified path
    data_path = args['data']['data_path']
    dataset = load_from_disk(data_path)

    # Sample subset if specified
    if args['data']['sample_subset']:
        total_size = len(dataset)
        sample_size = min(args['data']['sample_size'], total_size)
        rng = random.Random(args['data']['random_state'])
        sampled_indices = rng.sample(range(total_size), sample_size)
        dataset = dataset.select(sampled_indices)

    # Prepare for dataset split
    indices = list(range(len(dataset)))

    # Prepare train-test split parameters
    split_params = {
        'test_size': args['data']['split'],
        'random_state': args['data']['random_state']
    }
    if args['data']['stratify']:
        split_params['stratify'] = dataset['cell_types']

    # Perform train test split
    if args['data']['split'] > 0:
        train_indices, test_indices = train_test_split(indices, **split_params)

        if split_dataset:
            train_dataset = dataset.select(train_indices)
            test_dataset = dataset.select(test_indices)

            return train_dataset, test_dataset

        else:
            split_labels = {i: 'train' for i in train_indices}
            split_labels.update({i: 'test' for i in test_indices})
            def add_split_label(example, idx):
                return {'split': split_labels[idx]}
            dataset = dataset.map(add_split_label, with_indices=True)

            return dataset
            
