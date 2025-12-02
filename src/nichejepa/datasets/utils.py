import json
import pickle
import random
import requests
from typing import List, Literal, Tuple, Union

import datasets
from datasets import load_from_disk
from sklearn.model_selection import train_test_split

from ..utils.embedding import collect_adata_from_folder


def get_ensembl_ids(gene_names: List,
                    species: Literal['homo_sapiens',
                                     'mus_musculus'],
                    ) -> dict:
    """
    Get gene Ensembl IDs based on gene names via Ensembl REST API.

    Parameters
    ----------
    gene_names:
        List of gene names.
    species:
        Species for which to retrieve Ensembl IDs.

    Returns
    ----------
    ensembl_ids:
        Dictionary where keys are gene names and values are Ensembl IDs.
    """
    server = 'https://rest.ensembl.org'
    endpoint = f'/lookup/symbol/{species}'
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

    data = {'symbols': gene_names}
    response = requests.post(f'{server}{endpoint}',
                             headers=headers,
                             data=json.dumps(data))
    
    if response.ok:
        ensembl_ids = {}
        for key, value in response.json().items():
            ensembl_ids[key] = value['id']
        if len(ensembl_ids.keys()) != len(gene_names):
            missing_genes = [
                gene for gene in gene_names if gene not in ensembl_ids.keys()]
            print(f'Could not find Ensembl IDs for genes: {missing_genes}.')
        return ensembl_ids
    else:
        response.raise_for_status()


def prepare_dataset(args: dict,
                    split_dataset: bool=True,
                    ) -> Union[Tuple[datasets.Dataset, datasets.Dataset],
                               datasets.Dataset]:
    """
    Prepare dataset by loading it, determining sample size, and splitting it
    into training and test sets based on `split_dataset`.

    Parameters
    -----------
    args:
        A dictionary containing the configuration parameters, including:
            - tokenized_data_folder_path: The path to the tokenized dataset.
            - sample_subset: Whether to sample a subset of the dataset.
            - sample_size: The size of the dataset to sample.
            - split: The train-test split ratio.
            - stratify: Whether to stratify the dataset based on cell types
                        during the split.
            - random_state: The random seed for reproducibility.
    split_dataset:
        If 'True', split the dataset into train and test datasets.

    Returns
    -----------
    if `split_dataset` is True:
        train_dataset:
            The training portion of the dataset.
        test_dataset:
            The test portion of the dataset.
    else:
        dataset:
            The combined training and test portion of the dataset with a `split`
            label.
    """
    # Load dataset from the specified path
    data_path = args['data']['tokenized_data_folder_path']
    dataset = load_from_disk(data_path)

    if args['data'].get('precomputed_split'):
        with open(args['data']['precomputed_split'], 'rb') as f:
            indices = pickle.load(f)
        dataset = dataset.select(indices)
        if args['data'].get('chunk_dataset'):
            chunk_size = 12508671
            chunks = []
            for start in range(0, len(dataset), chunk_size):
                end = min(start + chunk_size, len(dataset))
                indices = list(range(start, end))
                chunk = dataset.select(indices)
                chunks.append(chunk)
            return chunks, None
        else:
            return dataset, dataset

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
    split_params = {'test_size': args['data']['split'],
                    'random_state': args['data']['random_state']}
    if args['data']['stratify']:
        split_params['stratify'] = dataset['cell_types']

    # Perform train test split
    if args['data']['split'] > 0:
        train_indices, test_indices = train_test_split(indices, **split_params)
    elif args['data']['test_batch_ids']:
        cell_ids = list(dataset['cell_id'])
        test_batch_mask = [
            any(batch_id in cell_id for batch_id in args['data'][
                'test_batch_ids']) for cell_id in cell_ids]
        train_indices = [
            index for index, value in enumerate(test_batch_mask) if not value]
        test_indices = [
            index for index, value in enumerate(test_batch_mask) if value]
    else:
        train_indices = indices
        test_indices = indices

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
