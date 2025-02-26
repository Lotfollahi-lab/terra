import json
import random
import requests
from typing import List, Literal, Tuple, Union

import pickle
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

    Returns
    -----------
    Either:
        dataset:
            The precomputed split of the dataset.
    or:
        train_dataset:
            The training portion of the dataset.
        test_dataset:
            The test portion of the dataset.
        val_dataset:
            The validation portion of the dataset.
    """
    # Load dataset from the specified path
    data_path = args['data']['tokenized_data_folder_path']
    dataset = load_from_disk(data_path)

    if args['data']['precomputed_split']:
        # Load precomputed data split if specified
        with open(args['data']['precomputed_split'], "rb") as f: 
            indices = pickle.load(f)
        dataset = dataset.select(indices)
        return dataset, None, None
    else:
        # Prepare for dataset split
        indices = list(range(len(dataset)))
        cell_ids = dataset['cell_id']

        if args['data']['test_batch_ids']:
            test_batch_mask = [
                any(batch_id == f"{cell_id.split('_')[0]}_{cell_id.split('_')[1]}"
                    for batch_id in args['data']['test_batch_ids'])
                for cell_id in cell_ids]
            test_indices = [
                index for index, value in enumerate(test_batch_mask) if value]
            train_indices = [
                index for index, value in enumerate(test_batch_mask) if not value]
        else:
            test_indices = []
            train_indices = indices

        if args['data']['val_batch_ids']:
            val_batch_mask = [
                any(batch_id == f"{cell_id.split('_')[0]}_{cell_id.split('_')[1]}"
                    for batch_id in args['data']['val_batch_ids'])
                for cell_id in cell_ids]
            val_indices = [
                index for index, value in enumerate(val_batch_mask) if value]
            train_indices = [
                index_1 and index_2 for index_1, index_2 in zip(
                    train_indices,
                    [index for index, value in enumerate(val_batch_mask) if not value]
                    )]
        else:
            val_indices = []

    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    test_dataset = dataset.select(test_indices)

    return train_dataset, val_dataset, test_dataset