"""
Cell Tokenizers.

Adapted from Theodoris, C. V. et al. Transfer learning enables predictions in
network biology. Nature 618, 616–624 (2023);
https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/tokenizer.py
(12.04.2024).

Input Data
----------
Required format:
    Raw counts spatial transcriptomics (ST) data with all genes (no feature
    selection) as '.h5ad' (anndata) files. Spatial coordinates are stored in
    adata.obsm['spatial'].
Required gene attributes:
    Ensembl ID for each gene ('ensembl_id').
Optional cell attributes:
    Binary indicator of whether cell should be used for tokenization based on
    user-defined filtering criteria ('filter_pass'). Any other cell metadata can
    be passed on to the tokenized dataset as a custom attribute dictionary.

Usage
----------
.. code-block :: python
    >>> from nichejepa import CellGraphRankTokenizer
    >>> tk = CellGraphRankTokenizer(nproc=4)
    >>> tk.tokenize_data(
    >>>     'input_directory', 'output_directory', 'output_file_prefix')

or

.. code-block :: python
    >>> from nichejepa import CellNeighborhoodRankTokenizer
    >>> tk = CellNeighborhoodRankTokenizer(nproc=4)
    >>> tk.tokenize_data(
    >>>     'input_directory', 'output_directory', 'output_file_prefix')

Description
----------
Input data is a directory with '.h5ad' files containing raw counts from ST data,
including all genes detected without feature selection. The input file type is
specified by the argument 'file_format' in the tokenize_data function. Genes
should be labeled with Ensembl IDs (adata.var['ensembl_id']), which provide a
unique identifer for conversion to tokens. Gene names can be converted to
Ensembl IDs via the helper function nichejepa.utils.genes.get_ensembl_ids() or
via the pyensembl Python package. No cell metadata is required, but custom cell
attributes may be passed onto the tokenized dataset by providing a dictionary of
custom attributes, which is formatted as {original_attr_name:
desired_dataset_attr_name}. For example, if the original '.h5ad' file has cell
attributes in adata.obs['cell_type'] and one would like to retain these
attributes as labels in the tokenized dataset with the new names 'cell_types',
the following custom attribute dictionary should be provided: {'cell_type':
'cell_types'}. Additionally, if the original '.h5ad' file contains a cell
attribute called adata.obs['filter_pass'], this will be used as a binary
indicator of whether to include these cells in the tokenization. All cells with
'1' in this attribute will be tokenized, whereas the others will be excluded.
One may use this column to indicate QC filtering or other criteria for selection
for inclusion in the final tokenized dataset. If one's data is in other formats
besides '.h5ad', one can use the relevant tools (such as Anndata tools) to
convert the file to '.h5ad' format prior to initializing the cell tokenizer.
"""

from __future__ import annotations

import logging
import pickle
import warnings
import concurrent
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import anndata as ad
import numpy as np
import scipy.sparse as sp
import squidpy as sq
from datasets import Dataset

from ..preprocessors.aggregators import aggregate_neighbors
from ..preprocessors.filters import filter_poor_quality_cells
from ..preprocessors.normalizers import normalize_by_analytic_pearson_residuals
from ..preprocessors.normalizers import normalize_by_cell_area
from ..preprocessors.normalizers import normalize_by_mean
from ..preprocessors.normalizers import normalize_by_nonzero_mean
from ..preprocessors.normalizers import normalize_by_read_depth
from ..preprocessors.normalizers import normalize_by_seurat
from ..preprocessors.normalizers import normalize_by_shifted_log_mean
from ..preprocessors.normalizers import normalize_by_shifted_log
from .tokenize import process_gene_expr, process_gene_tokens, rank_gene_tokens


warnings.filterwarnings('ignore', message=".*The 'nopython' keyword.*") # noqa
logger = logging.getLogger(__name__)


base_path = Path(__file__).parent.parent.parent.parent
CELL_GENE_MEANS_FILE = base_path / 'cell_gene_means_dictionary.pkl'
CELL_GENE_NZMEANS_FILE = base_path / 'cell_gene_nzmeans_dictionary.pkl'
CELL_GENE_LOGMEANS_FILE = base_path / 'cell_gene_logmeans_dictionary.pkl'
NEIGHBORHOOD_GENE_MEANS_FILE = base_path / 'neighborhood_gene_means_dictionary.pkl'
NEIGHBORHOOD_GENE_NZMEANS_FILE = base_path / 'neighborhood_gene_nzmeans_dictionary.pkl'
NEIGHBORHOOD_GENE_LOGMEANS_FILE = base_path / 'neighborhood_gene_logmeans_dictionary.pkl'
TOKEN_DICTIONARY_FILE = base_path / 'token_dictionary.pkl'
GENE_PANEL_ID_TO_GENE_PANEL_DICT_FILE = base_path / 'gene_panel_ID_to_gene_panel_dict.pkl'
FILE_PATH_TO_GENE_PANEL_ID_DICT_FILE = base_path / 'file_path_to_gene_panel_ID_dict.pkl'


class CellBaseTokenizer(ABC):
    def __init__(
        self,
        nproc: int=1,
        processing_mode: Optional[Literal[
            'sequential', 'parallel']]='sequential',
        chunk_size: int=512,
        model_input_size: int=2048,
        norm_factor: Optional[Literal["read_depth", "cell_area"]]=None,
        norm_method: Optional[Literal["analytic_pearson_residuals",
                                      "mean",
                                      "nzmean",
                                      "seurat_v3",
                                      "shifted_logmean"
                                      "shifted_log"]]="shifted_log",
        cell_gene_means_file: Path | str=CELL_GENE_MEANS_FILE,
        cell_gene_nzmeans_file: Path | str=CELL_GENE_NZMEANS_FILE,
        cell_gene_logmeans_file: Path | str=CELL_GENE_LOGMEANS_FILE,
        neighborhood_gene_means_file: Path | str=NEIGHBORHOOD_GENE_MEANS_FILE,
        neighborhood_gene_nzmeans_file: Path | str=NEIGHBORHOOD_GENE_NZMEANS_FILE,
        neighborhood_gene_logmeans_file: Path | str=NEIGHBORHOOD_GENE_LOGMEANS_FILE,
        token_dictionary_file: Path | str=TOKEN_DICTIONARY_FILE,
        gene_panel_ID_to_gene_panel_dict_file: Path | str=GENE_PANEL_ID_TO_GENE_PANEL_DICT_FILE,
        file_path_to_gene_panel_ID_dict_file: Path | str=FILE_PATH_TO_GENE_PANEL_ID_DICT_FILE,
        ):
        """
        CellBaseTokenizer class.
        """
        self.nproc = nproc
        self.processing_mode = processing_mode
        self.chunk_size = chunk_size
        self.model_input_size = model_input_size
        self.norm_factor = norm_factor
        self.norm_method = norm_method
        self.cell_gene_means_file = cell_gene_means_file
        self.cell_gene_nzmeans_file = cell_gene_nzmeans_file
        self.cell_gene_logmeans_file = cell_gene_logmeans_file
        self.neighborhood_gene_means_file = neighborhood_gene_means_file
        self.neighborhood_gene_nzmeans_file = neighborhood_gene_nzmeans_file
        self.neighborhood_gene_logmeans_file = neighborhood_gene_logmeans_file
        self.gene_panel_ID_to_gene_panel_dict_file = gene_panel_ID_to_gene_panel_dict_file
        self.file_path_to_gene_panel_ID_dict_file = file_path_to_gene_panel_ID_dict_file

        # Load token dictionary
        with open(token_dictionary_file, 'rb') as f:
            self.token_dict = pickle.load(f)

        # Load gene panel ID to gene panel dictionary
        logger.info('Loading gene panel ID to gene panel dictionary from '
                    f'{self.gene_panel_ID_to_gene_panel_dict_file}.')
        with open(self.gene_panel_ID_to_gene_panel_dict_file, 'rb') as f:
            self.gene_panel_ID_to_gene_panel_dict = pickle.load(f)
            
        # Load dictionary of file paths to gene panel IDs
        logger.info('Loading file path to gene panel ID dictionary from '
                    f'{self.file_path_to_gene_panel_ID_dict_file}.')
        with open(self.file_path_to_gene_panel_ID_dict_file, 'rb') as f:
            self.file_path_to_gene_panel_ID_dict = pickle.load(f)
            
        # Get vocabulary and gene Ensembl IDs (protein-coding and miRNA genes)
        self.vocab = list(self.token_dict.keys())
        self.coding_miRNA_ids = [
            key for key in list(self.vocab) if 'ENS' in key]
        self.coding_miRNA_dict = dict(
            zip(self.coding_miRNA_ids, [True] * len(self.vocab)))

        self.special_token_keys = ['batch_token',
                                   'gene_panel_token',
                                   'assay_token',
                                   'species_token',
                                   'tissue_token']      

    def tokenize_data(self,
                      input_directory: Path | str,
                      output_directory: Path | str,
                      output_file_prefix: str,
                      file_format: Literal['h5ad']='h5ad',
                      use_generator: bool=False,
                      cache_directory_path: Optional[Path | str]=None,
                      num_shards: int=None,
                      keep_in_memory: bool=False,
                      ):
        """
        Tokenize files in 'input_directory' and save as tokenized '.dataset'
        file in 'output_directory'.

        Parameters
        ----------
        input_directory:
            Path to directory containing '.h5ad' (anndata) files.
        output_directory:
            Path to directory where tokenized data will be saved as '.dataset'
            file.
        output_file_prefix:
            Prefix for output file.
        file_format:
            Format of input files. Can be '.h5ad'.
        use_generator:
            If 'True', use generator for tokenization, else dict.
        cache_directory_path:
            If specified, cache directory path for dataset creation.
        num_shards:
            Number of shards to save dataset to.
        keep_in_memory:
            If 'True', keep dataset in memory when using generator.
        """
        dataset_dict = self._tokenize_files(Path(input_directory), file_format)

        tokenized_dataset = self._create_dataset(
            dataset_dict=dataset_dict,
            use_generator=use_generator,
            cache_directory_path=cache_directory_path,
            keep_in_memory=keep_in_memory)

        output_path = str(
            (Path(output_directory) / output_file_prefix).with_suffix(
                '.dataset'))
        tokenized_dataset.save_to_disk(output_path, num_shards=num_shards)
        logger.info("Tokenized dataset saved to '{output_path}'.")

    def _create_dataset(self,
                        dataset_dict: dict,
                        use_generator: bool=False,
                        cache_directory_path: Optional[Path | str]=None,
                        keep_in_memory: bool=False,
                        ) -> Dataset:
        """
        Create a huggingface dataset based on tokenized cells.

        Parameters
        ----------
        dataset_dict:
            Dictionary based on which the huggingface dataset will be created.
        use_generator:
            If 'True', use generator for tokenization, else dict.
        cache_directory_path:
            If specified, cache directory path for dataset creation.
        keep_in_memory:
            If 'True', keep dataset in memory when using generator.

        Returns
        ----------
        dataset:
            Huggingface dataset containing the tokenized cells.
        """
        # Create huggingface dataset
        logger.info('Creating huggingface dataset...')
        if use_generator:
            def dict_generator():
                for i in range(len(dataset_dict['cell_id'])):
                    yield {k: dataset_dict[k][i] for k in dataset_dict.keys()}
            logger.info('Using generator for dataset creation.')
            dataset = Dataset.from_generator(dict_generator,
                                             num_proc=self.nproc,
                                             keep_in_memory=keep_in_memory,
                                             cache_dir=cache_directory_path)
        else:
            logger.info('Using dictionary for dataset creation.')
            dataset = Dataset.from_dict(dataset_dict)

        logger.info('Formatting gene tokens...')

        formatted_dataset = dataset.map(
            self._format_examples, 
            num_proc=self.nproc,
            keep_in_memory=keep_in_memory)
                
        return formatted_dataset

    def _tokenize_files(self,
                       data_directory: Path | str,
                       file_format: Literal['h5ad']='h5ad',
                       ) -> dict:
        """
        Tokenize multiple files in a directory.

        Parameters
        ----------
        data_directory:
            Path to the directory containing the files to be tokenized.
        file_format:
            Format of the files to be tokenized.

        Returns
        ----------
        dataset_dict:
            Dictionary containing the cell IDs and tokens for the tokenized
            files with the following keys and values:
                - cell_IDs: List of cell IDs.
                - gene_tokens_cell: Cell-wise vector of ranked cell gene tokens.
                - gene_tokens_neighborhood: Cell-wise vector of ranked
                      neighborhood gene tokens.
                - metadata columns: cell-wise metadata
                - special token columns: vectors of special tokens
        """
        file_found = 0

        tokenize_file_fn = self._tokenize_adata # add support of other file
                                                # formats in the future

        # Initialize dict to add results from individual files
        dataset_dict = {}

        # Loop through data directory to tokenize '.h5ad' files
        if self.processing_mode == 'sequential':
            logger.info('Tokenizing files sequentially...')
            for file_path in data_directory.glob(f'**/*.{file_format}'):
                file_found = 1
                logger.info(f"Tokenizing '{file_path}'.")
                file_dataset_dict = tokenize_file_fn(file_path)
                for k in file_dataset_dict.keys():
                    if k not in dataset_dict:
                        dataset_dict[k] = []
                    dataset_dict[k] += file_dataset_dict[k]
        elif self.processing_mode == 'parallel':
            logger.info('Tokenizing files in parallel...')
            with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.nproc) as executor:
                futures = []
                for file_path in data_directory.glob(f"**/*.{file_format}"):
                    file_found = 1
                    logger.info(f"Tokenizing '{file_path}'...")
                    future = executor.submit(tokenize_file_fn, file_path)
                    futures.append(future)
                for future in concurrent.futures.as_completed(futures):
                    file_dataset_dict = future.result()
                    for k in file_dataset_dict.keys():
                        if k not in dataset_dict:
                            dataset_dict[k] = []
                        dataset_dict[k] += file_dataset_dict[k]

        if file_found == 0:
            logger.error(
                f"No '.{file_format}' files found in directory "
                f"'{data_directory}'.")
            raise FileNotFoundError(
                f"No '.{file_format}' files found in directory "
                f"'{data_directory}'.")

        return dataset_dict

    @abstractmethod
    def _tokenize_adata(self):
        """
        Tokenizer-specific logic to tokenize one adata file.
        """
        pass

    @abstractmethod
    def _format_examples(self):
        """
        Tokenizer-specific logic to format examples.
        """
        pass


class CellGraphTokenizer(CellBaseTokenizer):
    pass


class CellNeighborhoodTokenizer(CellBaseTokenizer):
    def __init__(self,
                 **base_tokenizer_kwargs,
                 ):
        """
        CellNeighborhoodTokenizer class.

        Parameters
        -----------
        **base_tokenizer_kwargs:
            Keyword arguments for the initialization of CellBaseTokenizer.
        """
        super().__init__(**base_tokenizer_kwargs)

    def _tokenize_adata(self,
                        adata_file_path: Path | str
                        ) -> dict:
        """
        Tokenize cells from an '.h5ad' (anndata) file.

        Parameters
        ----------
        adata_file_path:
            Path to anndata file containing cells to be tokenized.

        Returns
        ----------
        adata_dict:
            Dictionary with keys:
            - gene_tokens_cell:
                Cell-wise vector of ranked cell gene tokens.
            - gene_tokens_neighborhood:
                Cell-wise vector of ranked neighborhood gene tokens.
            cell_metadata:
                Dictionary of cell metadata where keys are metadata columns and
                values are lists of cell-wise values.
            special_tokens_dict:
                Dictionary of special tokens where keys are special token names and
                values are special token values (for each cell).
            cell_IDs:
                List of cell IDs.
        """
        # Initialize dict to collect tokens and cell ids
        adata_dict = {}

        adata = ad.read_h5ad(adata_file_path)

        print('Filtering cells.')
        # Filter to remove poor quality cells
        adata = filter_poor_quality_cells(adata)

        print('Computing spatial neighborhood graph and aggregating counts.')
        # Aggregate neighborhood cell gene expression
        adata = aggregate_neighbors(adata,
                                    radius=27.5)

        print('Normalizing gene expression counts.')
        # Perform normalization of total counts per cell
        if self.norm_factor == 'read_depth':
            if self.norm_method == 'analytic_pearson_residuals':
                raise ValueError(
                    "Invalid combination of 'norm_factor' and 'norm_method': "
                    f"({self.norm_factor} / {self.norm_method}).")
            adata.X = normalize_by_read_depth(adata.X)
            adata.layers['X_neighborhood'] = normalize_by_read_depth(
                adata.layers['X_neighborhood'])
        elif self.norm_factor == 'cell_area':
            if self.norm_method == 'analytic_pearson_residuals':
                raise ValueError(
                    "Invalid combination of 'norm_factor' and 'norm_method': "
                    f"({self.norm_factor} / {self.norm_method}).")
            adata.X = normalize_by_cell_area(
                adata.X,
                cell_areas=adata.obs['cell_area'].values)
            adata.obs['neighborhood_cell_area'] = np.array(
                adata.obsp['spatial_connectivities'].T @
                adata.obs['cell_area'].values.reshape(-1, 1))
            adata.X = normalize_by_cell_area(
                adata.layers['X_neighborhood'],
                cell_areas=adata.obs['neighborhood_cell_area'].values)
        else:
            if self.norm_factor is None:
                pass
            else:
                raise ValueError(f"Invalid 'norm_factor' {self.norm_factor}.")
                
        # Perform normalization of counts
        if self.norm_method == 'analytic_pearson_residuals':
            adata.X = normalize_by_analytic_pearson_residuals(adata.X)
            adata.layers['X_neighborhood'] = normalize_by_analytic_pearson_residuals(
                adata.layers['X_neighborhood'])
        elif self.norm_method == 'seurat_v3':
            adata.X = normalize_by_seurat(adata.X)
            adata.layers['X_neighborhood'] = normalize_by_seurat(
                adata.layers['X_neighborhood'])
        elif self.norm_method == 'mean':
            adata.X = normalize_by_mean(
                adata.X,
                gene_means_file=self.cell_gene_means_file,
                probed_genes=adata.var['ensembl_id'])
            adata.layers['X_neighborhood'] = normalize_by_mean(
                adata.layers['X_neighborhood'],
                gene_means_file=self.neighborhood_gene_means_file,
                probed_genes=adata.var['ensembl_id'])
        elif self.norm_method == 'nzmean':
            adata.X = normalize_by_nonzero_mean(
                adata.X,
                gene_nzmeans_file=self.cell_gene_nzmeans_file,
                probed_genes=adata.var['ensembl_id'])
            adata.layers['X_neighborhood'] = normalize_by_nonzero_mean(
                adata.layers['X_neighborhood'],
                gene_nzmeans_file=self.neighborhood_gene_nzmeans_file,
                probed_genes=adata.var['ensembl_id'])
        elif self.norm_method == 'shifted_logmean':
            adata.X = normalize_by_shifted_log_mean(
                adata.X,
                gene_logmeans_file=self.cell_gene_logmeans_file,
                probed_genes=adata.var['ensembl_id'])
            adata.layers['X_neighborhood'] = normalize_by_shifted_log_mean(
                adata.layers['X_neighborhood'],
                gene_logmeans_file=self.neighborhood_gene_logmeans_file,
                probed_genes=adata.var['ensembl_id'])
        elif self.norm_method == 'shifted_log':
            adata.X = normalize_by_shifted_log(adata.X)
            adata.layers['X_neighborhood'] = normalize_by_shifted_log(
                adata.layers['X_neighborhood'])
        else:
            if self.norm_method is None:
                pass
            else:
                raise ValueError(f"Invalid 'norm_method': {self.norm_method}.")

        # Initialize dict to collect tokens and cell IDs
        adata_dict = {}

        # Retrieve gene tokens for genes contained in dataset and vocab, i.e.
        # protein-coding and miRNA genes
        print('Retrieving gene tokens.')
        coding_miRNA_idx = np.where(
            [self.coding_miRNA_dict.get(
                gene_id, False) for gene_id in adata.var['ensembl_id']])[0]
        coding_miRNA_ids = adata.var['ensembl_id'][coding_miRNA_idx]

        coding_miRNA_tokens_cell = np.array(
            [self.token_dict[gene_id] for gene_id in coding_miRNA_ids])
        coding_miRNA_tokens_neighborhood = np.array(
            [self.token_dict[gene_id] for gene_id in coding_miRNA_ids])

        # Prepare gene tokens for cell and neighborhood for this file
        adata_dict['gene_tokens_cell'] = []
        adata_dict['gene_expr_cell'] = []
        adata_dict['gene_tokens_neighborhood'] = []
        adata_dict['gene_expr_neighborhood'] = []
            
        # Divide cells into chunks and loop through chunks
        print('Ranking gene tokens based on normalized counts.')
        for i in range(0, len(adata), self.chunk_size):
            norm_counts_cell = sp.csr_matrix(adata[
                i : i + self.chunk_size, coding_miRNA_idx].X)
            norm_counts_neighborhood = sp.csr_matrix(adata[
                i : i + self.chunk_size, coding_miRNA_idx].layers[
                    'X_neighborhood'])

            # Rank gene tokens and append across chunks
            adata_dict['gene_tokens_cell'] += [
                rank_gene_tokens(norm_counts_cell[j].data,
                coding_miRNA_tokens_cell[norm_counts_cell[j].indices])
                for j in range(norm_counts_cell.shape[0])]
            adata_dict['gene_tokens_neighborhood'] += [
                rank_gene_tokens(norm_counts_neighborhood[j].data,
                coding_miRNA_tokens_neighborhood[
                    norm_counts_neighborhood[j].indices])
                for j in range(norm_counts_neighborhood.shape[0])]

            # Append gene expression across chunks
            adata_dict['gene_expr_cell'] += [
                norm_counts_cell[j].data[np.argsort(-norm_counts_cell[j].data)]
                for j in range(norm_counts_cell.shape[0])]
            adata_dict['gene_expr_neighborhood'] += [
                norm_counts_neighborhood[j].data[
                    np.argsort(-norm_counts_neighborhood[j].data)]
                for j in range(norm_counts_neighborhood.shape[0])]

        # Add cell IDs for collecting metadata at inference time
        adata_dict['cell_id'] = adata.obs['cell_id'].values.tolist()
        
        # Add special tokens
        n_cells = len(adata)
        batch_id_key = f"{adata.uns['dataset_id']}_{adata.uns['batch']}"
        adata_dict['batch_token'] = [self.token_dict[batch_id_key]] * n_cells
        adata_dict['gene_panel_token'] = [
            self.token_dict[
                self.file_path_to_gene_panel_ID_dict[
                    str(adata_file_path)]]] * n_cells
        adata_dict['assay_token'] = [
            self.token_dict[adata.uns['assay']]] * n_cells
        adata_dict['species_token'] = [
            self.token_dict[adata.uns['species']]] * n_cells
        adata_dict['tissue_token'] = [
            self.token_dict[adata.uns['tissue']]] * n_cells

        return adata_dict
            
    def _format_examples(self,
                         example: dict) -> dict:
        """
        Format examples.
        """
        # Retrieve gene tokens
        gene_tokens_cell, n_nonzero_cell_tokens = process_gene_tokens(
            example['gene_tokens_cell'],
            int(self.model_input_size / 2),
            self.token_dict)
        del example['gene_tokens_cell']
        gene_tokens_neighborhood, n_nonzero_neighborhood_tokens = process_gene_tokens(
            example['gene_tokens_neighborhood'],
            int(self.model_input_size / 2),
            self.token_dict)
        del example['gene_tokens_neighborhood']
        example['gene_tokens'] = np.concatenate(
            (gene_tokens_cell.copy(), gene_tokens_neighborhood.copy()))

        # Retrieve gene expression
        gene_expr_cell = process_gene_expr(
            example['gene_expr_cell'],
            int(self.model_input_size / 2))
        del example['gene_expr_cell']
        gene_expr_neighborhood = process_gene_expr(
            example['gene_expr_neighborhood'],
            int(self.model_input_size / 2))
        del example['gene_expr_neighborhood']
        example['gene_expr'] = np.concatenate(
            (gene_expr_cell.copy(), gene_expr_neighborhood.copy()))

        # Retrieve attributes
        example['n_nonzero_tokens'] = (
            n_nonzero_cell_tokens + n_nonzero_neighborhood_tokens)

        # Define segments
        example['segments'] = np.concatenate(
            (np.array([1] * len(gene_tokens_cell)),
             np.array([2] * len(gene_tokens_neighborhood))))
        
        # Retrieve special tokens
        example['cls_cell_token'] = [self.token_dict['<cls_cell>']]
        example['cls_neighborhood_token'] = [
            self.token_dict['<cls_neighborhood>']]
        example['assay_token'] = [example['assay_token']]
        example['species_token'] = [example['species_token']]
        example['tissue_token'] = [example['tissue_token']]
        example['gene_panel_token'] = [example['gene_panel_token']]
        example['batch_token'] = [example['batch_token']]

        return example