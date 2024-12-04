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
    Ensembl ID for each gene (adata.var['ensembl_id']).
Required cell attributes:
    Cell ID in index. Metadata is retrieved at inference time via this cell ID.
Optional cell attributes:
    Binary indicator of whether cell should be used for tokenization based on
    user-defined filtering criteria (adata.obs['filter_pass']).

Usage
----------
.. code-block :: python
    >>> from nichejepa import CellGraphTokenizer
    >>> tk = CellGraphTokenizer(max_cores=4)
    >>> tk.tokenize_data(
    >>>     'input_directory', 'output_directory', 'output_file_prefix')

or

.. code-block :: python
    >>> from nichejepa import CellNeighborhoodTokenizer
    >>> tk = CellNeighborhoodTokenizer(max_cores=4)
    >>> tk.tokenize_data(
    >>>     'input_directory', 'output_directory', 'output_file_prefix')

Description
----------
Input data is a directory with '.h5ad' files containing raw counts from ST data,
including all genes detected without feature selection. The input file type is
specified by the argument 'file_format' in the tokenize_data function. Genes
should be labeled with Ensembl IDs (adata.var['ensembl_id']), which provide a
unique identifer for conversion to tokens. Gene names can be converted to
Ensembl IDs via the helper function nichejepa.datasets.utils.get_ensembl_ids()
or via the pyensembl Python package. No cell metadata is required, but the cell
ID needs to be stored in the index. Additionally, if the original '.h5ad' file
contains a cell attribute called adata.obs['filter_pass'], this will be used as
a binary indicator of whether to include these cells in the tokenization. All
cells with '1' in this attribute will be tokenized, whereas the others will be
excluded. One may use this column to indicate QC filtering or other criteria for
selection for inclusion in the final tokenized dataset. If one's data is in
other formats besides '.h5ad', one can use the relevant tools (such as Anndata
tools) to convert the file to '.h5ad' format prior to initializing the cell
tokenizer.
"""


from __future__ import annotations

import concurrent.futures
import shutil
import os
import concurrent
import logging
import pickle
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional
import tracemalloc


import anndata as ad
import numpy as np
import scipy.sparse as sp
from datasets import Dataset, concatenate_datasets
from datasets.formatting.formatting import LazyRow, LazyBatch

from ..preprocessors.aggregators import aggregate_neighbors
from ..preprocessors.filters import filter_poor_quality_cells
from ..preprocessors.normalizers import normalize_by_analytic_pearson_residuals
from ..preprocessors.normalizers import normalize_by_cell_area
from ..preprocessors.normalizers import normalize_by_mean
from ..preprocessors.normalizers import normalize_by_nonzero_mean
from ..preprocessors.normalizers import normalize_by_read_depth
from ..preprocessors.normalizers import normalize_by_seurat
from ..preprocessors.normalizers import normalize_by_shifted_log
from ..preprocessors.normalizers import normalize_by_shifted_log_mean
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

MAX_THREADS = 256


class CellBaseTokenizer(ABC):
    def __init__(
        self,
        max_cores: int=1,
        chunk_size: int=2048,
        model_input_size: int=2048,
        include_zero_expr_genes: bool=False,
        n_neighs: Optional[float]=None,
        radius: Optional[float]=None,
        delaunay: bool=True,
        norm_factor: Optional[Literal['read_depth', 'cell_area']]=None,
        norm_method: Optional[Literal['analytic_pearson_residuals',
                                      'mean',
                                      'nzmean',
                                      'seurat_v3',
                                      'shifted_logmean'
                                      'shifted_log']]='shifted_log',
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

        Parameters
        ----------
        n_neighs:
            If specified, use `n_neighs` to compute the neighborhood graph. If
            'radius' or 'delaunay' are also specified, a union neighborhood
            graph will be computed.
        radius:
            If specified, use `radius` to compute the neighborhood graph. If
            'n_neighs' or 'delaunay' are also specified, a union neighborhood
            graph will be computed.
        delaunay:
            If 'True', compute the neighborhood graph by delaunay triangulation.
            If 'n_neighs' or 'radius' are also specified, a union neighborhood
            graph will be computed.
        """
        self.max_cores = max_cores
        self.max_threads = min(MAX_THREADS, os.cpu_count())
        self.chunk_size = chunk_size
        self.model_input_size = model_input_size
        self.include_zero_expr_genes = include_zero_expr_genes
        self.n_neighs = n_neighs
        self.radius = radius
        self.delaunay = delaunay
        self.norm_factor = norm_factor
        self.norm_method = norm_method
        self.cell_gene_means_file = cell_gene_means_file
        self.cell_gene_nzmeans_file = cell_gene_nzmeans_file
        self.cell_gene_logmeans_file = cell_gene_logmeans_file
        self.neighborhood_gene_means_file = neighborhood_gene_means_file
        self.neighborhood_gene_nzmeans_file = neighborhood_gene_nzmeans_file
        self.neighborhood_gene_logmeans_file = neighborhood_gene_logmeans_file
        self.token_dictionary_file = token_dictionary_file
        self.gene_panel_ID_to_gene_panel_dict_file = gene_panel_ID_to_gene_panel_dict_file
        self.file_path_to_gene_panel_ID_dict_file = file_path_to_gene_panel_ID_dict_file

        # Load token dictionary
        logger.info('Loading token dictionary from '
                    f'{self.token_dictionary_file}.')
        with open(token_dictionary_file, 'rb') as f:
            self.token_dict = pickle.load(f)

        self.max_cls_tokens = sum(1 for key in self.token_dict if "cls" in key)
        self.max_special_tokens = self.max_cls_tokens + sum(
            1 for key in self.token_dict if "spt" in key)

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

    def tokenize_data(self,
                      input_directory: Path | str,
                      output_directory: Path | str,
                      output_file_prefix: str,
                      file_format: Literal['h5ad']='h5ad',
                      num_shards: int=None,
                      keep_in_memory: bool=False,
                      cache_directory_path: Optional[Path | str]=None,
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
        num_shards:
            Number of shards to save dataset to.
        keep_in_memory:
            If 'True', keep dataset in memory when using generator.
        cache_directory_path:
            If specified, cache directory path for dataset creation.
        """
        # create Hugging Face Dataset object from tokenized batches of cells
        dataset = self._tokenize_files(
                        Path(input_directory),
                        file_format,
                        keep_in_memory=keep_in_memory,
                        cache_directory_path=cache_directory_path,
                    )
        
        # format tokens
        formatted_dataset = self._format_dataset(
            dataset=dataset,
            keep_in_memory=keep_in_memory,
            )

        # save tokenized dataset to disk
        output_path = str(
            (Path(output_directory) / output_file_prefix).with_suffix(
                '.dataset'))
        formatted_dataset.save_to_disk(output_path, num_shards=num_shards)
        print(f"Tokenized dataset saved to {output_path}.")
        logger.info("Tokenized dataset saved to '{output_path}'.")


    def _format_dataset(self,
                        dataset: Dataset,
                        keep_in_memory: bool=False,
                        ) -> Dataset:
        """
        Format the tokens for each cell in the Hugging Face dataset.

        Parameters
        ----------
        dataset:
            Hugging Face dataset containing the tokenized cells.
        keep_in_memory:
            If 'True', keep dataset in memory when using generator.

        Returns
        ----------
        dataset:
            Hugging Face dataset containing the formatted tokens for the cells.
        """
        num_proc = min(self.max_threads, int(len(dataset)/self.chunk_size))
        use_batching = True if num_proc > 1 else False
        batch_size = self.chunk_size*2
        load_from_cache_file = False if keep_in_memory else True
        
        logger.info('Formatting gene tokens...')
        print(f"Parallelizing using {num_proc} processes and batch size of {batch_size}...")

        tracemalloc.start()
        formatted_dataset = dataset.map(
                        self._format_tokens_of_a_batch_of_cells,
                        num_proc=num_proc,
                        batched=use_batching, # by default, this is set to False
                        batch_size=batch_size, # by default, this is set to 1000, but we set it to 4096
                        load_from_cache_file=load_from_cache_file, # by default, this is set to True
                        keep_in_memory=keep_in_memory,
                    )
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Peak memory usage while Formatting Cell Tokens was {(peak / 10**6):.2f} MB.")
        print(f"Current memory usage is {(current / 10**6):.2f} MB.")
        
        print(f"Number of keys in unformatted dataset: {len(dataset[0].keys())}")
        print(f"Number of keys in formatted dataset: {len(formatted_dataset[0].keys())}")
        print(f"Keys in unformatted dataset: {dataset[0].keys()}")
        print(f"Keys in formatted dataset: {formatted_dataset[0].keys()}")
        extra_keys = set(formatted_dataset[0].keys()) - set(dataset[0].keys())
        print(f"Keys in formatted dataset that are not in dataset: {extra_keys}")
        extra_keys = set(dataset[0].keys()) - set(formatted_dataset[0].keys())
        print(f"Keys in unformatted dataset that are not in formatted dataset: {extra_keys}")
        # assert dataset[0]['gene_tokens'] == formatted_dataset[0]['gene_tokens']
        
        # clean up dataset cache because dataset is no longer used if cache is used
        # if no cache files exist, this will not do anything
        print(f"Cleaning up cache...")

        # list dataset-related arrow cache files
        for cache_file in dataset.cache_files:
            print(f"Cache file: {cache_file['filename']}")
        # remove .arrow and .lock cache files
        num_removed_files = dataset.cleanup_cache_files()
        print(f'{num_removed_files} cache files removed.')
        
        # remove cache directory
        try:
            cache_dir = Path(os.path.dirname(os.path.abspath(cache_file['filename']))).parent
            if cache_dir.exists() and cache_dir.is_dir():
                shutil.rmtree(cache_dir)
            print(f"Removed cache directory {cache_dir}...")
        except:
            print(f"No cache directory to remove...")
        
        return formatted_dataset


    def _format_tokens_of_a_batch_of_cells(self,
                                            unformatted_examples,
                                            ) -> dict:
        """
        Format the tokens for a single cell or a batch of cells.
        
        Parameters
        ----------
        unformatted_examples:
            LazyBatch of cells if batching is used and a LazyRow for a single cell if batching is not used.
            
        Returns
        ----------
        batch_examples:
            Dictionary containing the formatted tokens for the cell(s).
        """
        # examples = {'cell_id': ['cell1', 'cell2', ...],
        #            'gene_tokens_cell': [array([1, 2, 3, ...]), array([4, 5, 6, ...]), ...],
        #             ...
        #             }
        if isinstance(unformatted_examples, LazyBatch):
            curr_batch_size = len(unformatted_examples['cell_id'])
            keys = unformatted_examples.keys()

            formatted_examples = {}
            
            for i in range(curr_batch_size):
                unformatted_example = {}
                for key in keys:
                    unformatted_example[key] = unformatted_examples[key][i]
                    
                formatted_example = self._format_tokens_of_a_single_cell(unformatted_example)

                for k in formatted_example.keys():
                    if k not in formatted_examples:
                        formatted_examples[k] = []
                    formatted_examples[k].append(formatted_example[k])

            return formatted_examples
        
        elif isinstance(unformatted_examples, LazyRow):
            return self._format_tokens_of_a_single_cell(unformatted_examples)


    def _tokenize_files(self,
                        data_directory: Path | str,
                        file_format: Literal['h5ad']='h5ad',
                        keep_in_memory: bool=False,
                        cache_directory_path: Optional[Path | str]=None,                        
                        ) -> Dataset:
        """
        Tokenize multiple files in a directory.

        Parameters
        ----------
        data_directory:
            Path to the directory containing the files to be tokenized.
        file_format:
            Format of the files to be tokenized.
        keep_in_memory:
            If 'True', keep dataset in memory when using generator.
        cache_directory_path:
            If specified, cache directory path for dataset creation.

        Returns
        ----------
        Dataset:
            Hugging Face dataset containing the tokenized cells.
        """
        file_paths = list(data_directory.glob(f'**/*.{file_format}'))
        self.num_files = len(file_paths)

        if self.num_files == 0:
            logger.error(
                f"No '.{file_format}' files found in directory "
                f"'{data_directory}'.")
            raise FileNotFoundError(
                f"No '.{file_format}' files found in directory "
                f"'{data_directory}'.")

        tokenize_file_fn = self._tokenize_adata # add support of other file
                                                # formats in the future

        # Loop through data directory to tokenize '.h5ad' files
        if self.num_files == 1:
            print(f"Found 1 '.{file_format}' file in '{data_directory}'.")
            for file_path in data_directory.glob(f'**/*.{file_format}'):
                print(f"Tokenizing '{file_path}'...")
                dataset = tokenize_file_fn(
                                adata_file_path=file_path,
                                keep_in_memory=keep_in_memory,
                                cache_directory_path=cache_directory_path,
                                )
                
        elif self.num_files > 1:
            print(f"Found {self.num_files} '.{file_format}' files in "
                        f"'{data_directory}'.")            
            print(f"Tokenizing files in parallel using {self.max_cores} cores...")
            
            hf_datasets = []
            
            with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_cores) as executor:
                futures = []
                for file_path in data_directory.glob(f'**/*.{file_format}'):
                    logger.info(f"Tokenizing '{file_path}'...")
                    future = executor.submit(tokenize_file_fn,
                                             file_path,
                                             keep_in_memory,
                                             cache_directory_path,
                                             )
                    futures.append(future)
                for future in concurrent.futures.as_completed(futures):
                    hf_datasets.append(future.result())

            logger.info('Concatenating datasets...')
            dataset = concatenate_datasets(hf_datasets)
            
        return dataset


    @abstractmethod
    def _tokenize_adata(self):
        """
        Tokenizer-specific logic to tokenize one adata file.
        """
        pass

    @abstractmethod
    def _format_tokens_of_a_single_cell(self):
        """
        Tokenizer-specific logic to format examples.
        """
        pass


class CellGraphTokenizer(CellBaseTokenizer):
    def __init__(self,
                 **base_tokenizer_kwargs,
                 ):
        """
        CellGraphTokenizer class.

        Parameters
        -----------
        **base_tokenizer_kwargs:
            Keyword arguments for the initialization of the CellBaseTokenizer.
        """
        super().__init__(**base_tokenizer_kwargs)

    def _tokenize_adata(self,
                        adata_file_path: Path | str,
                        keep_in_memory: bool=False,
                        cache_directory_path: Optional[Path | str]=None,
                        ) -> dict:
        """
        Tokenize cells from an '.h5ad' (anndata) file.

        Parameters
        ----------
        adata_file_path:
            Path to anndata file containing cells to be tokenized.
        keep_in_memory:
            If 'True', keep dataset in memory when using generator.
        cache_directory_path:
            If specified, cache directory path for dataset creation.

        Returns
        ----------
        adata_dict:
            Dictionary with tokenized data stored in keys:
            - gene_tokens_cell:
                Cell-wise vector of ranked cell gene tokens.
            - gene_expr_cell:
                Cell-wise vector of ranked cell gene expression.
            - gene_tokens_neighborhood:
                Cell-wise vector of ranked neighborhood gene tokens.
            - gene_expr_neighborhood:
                Cell-wise vector of ranked neighborhood gene expression.
            - seg_tokens_neighborhood:
                Segment tokens for the neighborhood (each neighbor cell is a
                different segment).
            - assay_token:
                List containing assay token.
            - species_tokens:
                List containing species token.
            - tissue_token:
                List containing tissue token.
            - gene_panel_token:
                List containing gene panel token.
            - batch_token:
                List containing batch token.
            - cell_ids:
                List of cell IDs.
        """
        # Initialize dict to collect tokens and cell ids
        adata_dict = {}

        adata = ad.read_h5ad(adata_file_path)

        print('Filtering cells.')
        # Filter to remove poor quality cells
        adata = filter_poor_quality_cells(adata)

        assert np.all(np.mod(adata.X.data, 1) == 0), "adata.X contains non-integer values."

        print('Computing spatial neighborhood graph.')
        adata = aggregate_neighbors(
            adata,
            n_neighs=self.n_neighs,
            radius=self.radius,
            delaunay=self.delaunay,
            include_self_loop=False)

        print('Normalizing gene expression counts.')
        # Perform normalization of total counts per cell
        if self.norm_factor == 'read_depth':
            if self.norm_method == 'analytic_pearson_residuals':
                raise ValueError(
                    "Invalid combination of 'norm_factor' and 'norm_method': "
                    f'({self.norm_factor} / {self.norm_method}).')
            adata.X = normalize_by_read_depth(adata.X)

        elif self.norm_factor == 'cell_area':
            if self.norm_method == 'analytic_pearson_residuals':
                raise ValueError(
                    "Invalid combination of 'norm_factor' and 'norm_method': "
                    f'({self.norm_factor} / {self.norm_method}).')
            adata.X = normalize_by_cell_area(
                adata.X,
                cell_areas=adata.obs['cell_area'].values)
        else:
            if self.norm_factor is None:
                pass
            else:
                raise ValueError(f"Invalid 'norm_factor' {self.norm_factor}.")
                
        # Perform normalization of counts
        if self.norm_method == 'analytic_pearson_residuals':
            adata.X = normalize_by_analytic_pearson_residuals(adata.X)
        elif self.norm_method == 'seurat_v3':
            adata.X = normalize_by_seurat(adata.X)
        elif self.norm_method == 'mean':
            adata.X = normalize_by_mean(
                adata.X,
                gene_means_file=self.cell_gene_means_file,
                probed_genes=adata.var['ensembl_id'])
        elif self.norm_method == 'nzmean':
            adata.X = normalize_by_nonzero_mean(
                adata.X,
                gene_nzmeans_file=self.cell_gene_nzmeans_file,
                probed_genes=adata.var['ensembl_id'])
        elif self.norm_method == 'shifted_logmean':
            adata.X = normalize_by_shifted_log_mean(
                adata.X,
                gene_logmeans_file=self.cell_gene_logmeans_file,
                probed_genes=adata.var['ensembl_id'])
        elif self.norm_method == 'shifted_log':
            adata.X = normalize_by_shifted_log(adata.X)
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

        # Prepare gene tokens for cell and neighborhood for this file
        adata_dict['gene_tokens_cell'] = []
        adata_dict['gene_expr_cell'] = []
        adata_dict['gene_tokens_neighborhood'] = []
        adata_dict['gene_expr_neighborhood'] = []
            
        # Divide cells into chunks and loop through chunks
        print('Ranking gene tokens based on normalized counts.')
        for i in range(0, len(adata), self.chunk_size):
            if self.include_zero_expr_genes:
                norm_counts_cell = adata[
                    i : i + self.chunk_size, coding_miRNA_idx].X.toarray()

                # Rank gene tokens and append across chunks
                adata_dict['gene_tokens_cell'] += [
                    rank_gene_tokens(norm_counts_cell[j],
                    coding_miRNA_tokens_cell)
                    for j in range(norm_counts_cell.shape[0])]

                # Rank gene expression and append across chunks
                adata_dict['gene_expr_cell'] += [
                    norm_counts_cell[j][np.argsort(-norm_counts_cell[j])]
                    for j in range(norm_counts_cell.shape[0])]
            else:
                norm_counts_cell = sp.csr_matrix(adata[
                    i : i + self.chunk_size, coding_miRNA_idx].X)

                # Rank gene tokens and append across chunks
                adata_dict['gene_tokens_cell'] += [
                    rank_gene_tokens(norm_counts_cell[j].data,
                    coding_miRNA_tokens_cell[norm_counts_cell[j].indices])
                    for j in range(norm_counts_cell.shape[0])]

                # Rank gene expression and append across chunks
                adata_dict['gene_expr_cell'] += [
                    norm_counts_cell[j].data[np.argsort(-norm_counts_cell[j].data)]
                    for j in range(norm_counts_cell.shape[0])]

        print('Retrieving tokens for neighborhood cells.')
        adata_dict['gene_tokens_neighborhood'] = [
            np.array([]) for i in range(len(adata))]
        adata_dict['gene_expr_neighborhood'] = [
            np.array([]) for i in range(len(adata))]
        adata_dict['seg_tokens_neighborhood'] = [
            np.array([]) for i in range(len(adata))]
        
        adata_dict['cell_degrees'] = []
        
        # Loop through all cells to add neighbor cell gene tokens based on
        # position of neighbor cell compared to index cell. Gene tokens of cells
        # that are closer to the index cell will be added first.
        for i in range(len(adata)):
            # Collect all neighbors of cell i (excluding cell i)
            neighbors_i = adata.obsp['spatial_connectivities'][i].nonzero()[1]

            # Store cell degree (number of neighbors excluding cell i)
            adata_dict['cell_degrees'].append(int(len(neighbors_i)))

            # Take into account case where neighbor cells can have 0 distance
            if (adata.obsp['spatial_connectivities'].getnnz(axis=1)[i] != 
            adata.obsp['spatial_distances'].getnnz(axis=1)[i]):
                cell_con_nz = np.nonzero(
                    adata.obsp['spatial_connectivities'][i])[1]
                cell_dist_nz = np.nonzero(
                    adata.obsp['spatial_distances'][i])[1]
                zero_dist_idx = set(cell_con_nz) - set(cell_dist_nz)
                for idx in zero_dist_idx:
                    adata.obsp['spatial_distances'][i, idx] = adata.obsp[
                        'spatial_distances'][i, idx] + 10**(-9)
            
            # Get sorted indices of neighbor cells based on (lower) distance to
            # index cell
            cell_start = adata.obsp['spatial_distances'].indptr[i]
            cell_end = adata.obsp['spatial_distances'].indptr[i+1]
            cell_distances = adata.obsp[
                'spatial_distances'].data[cell_start:cell_end]

            sorted_indices = np.argsort(cell_distances)
            assert len(neighbors_i) == len(cell_distances), (
                'Number of neighbors (excluding cell i) does not equal number '
                'of distances.')

            # Loop through distance-sorted neighbor cells and add gene and
            # segment tokens and counts
            for j, k in enumerate(neighbors_i[sorted_indices]):
                adata_dict['gene_tokens_neighborhood'][i] = np.hstack(
                    (adata_dict['gene_tokens_neighborhood'][i],
                     adata_dict['gene_tokens_cell'][k]))
                adata_dict['gene_expr_neighborhood'][i] = np.hstack(
                    (adata_dict['gene_expr_neighborhood'][i],
                     adata_dict['gene_expr_cell'][k]))
                adata_dict['seg_tokens_neighborhood'][i] = np.hstack(
                    (adata_dict['seg_tokens_neighborhood'][i],
                     [j + self.max_special_tokens + 1] * len(
                        adata_dict['gene_tokens_cell'][k])))

        # Add cell IDs for collecting metadata at inference time
        adata_dict['cell_id'] = adata.obs['cell_id'].values.tolist()
        
        # Add special tokens (positional tokens and value tokens)
        n_cells = len(adata)
        batch_id_key = f"{adata.uns['dataset_id']}_{adata.uns['batch']}"

        adata_dict['batch_token'] = [self.token_dict['spt_batch']] * n_cells
        adata_dict['gene_panel_token'] = [
            self.token_dict['spt_gene_panel']] * n_cells
        adata_dict['assay_token'] = [self.token_dict['spt_assay']] * n_cells
        adata_dict['species_token'] = [self.token_dict['spt_species']] * n_cells
        adata_dict['tissue_token'] = [self.token_dict['spt_tissue']] * n_cells

        adata_dict['batch_value_token'] = [
            self.token_dict[f'spv_{batch_id_key}']] * n_cells
        adata_dict['gene_panel_value_token'] = [self.token_dict[
            f'spv_{self.file_path_to_gene_panel_ID_dict[str(adata_file_path)]}']
            ] * n_cells
        adata_dict['assay_value_token'] = [
            self.token_dict[f'spv_{adata.uns["assay"]}']] * n_cells
        adata_dict['species_value_token'] = [
            self.token_dict[f'spv_{adata.uns["species"]}']] * n_cells
        adata_dict['tissue_value_token'] = [
            self.token_dict[f'spv_{adata.uns["tissue"]}']] * n_cells

        # Store values with right embedding index for count tokenizer
        # Leave space for <pad>, (optional) zero count embedding, and
        # <cls> tokens
        spv_dict = {
            k: v for k, v in self.token_dict.items() if k.startswith('spv_')}
        spv_start_idx = min(spv_dict.values())
        spv_idx_subtract = spv_start_idx - 2 - self.max_cls_tokens
        adata_dict['batch_value'] = [
            self.token_dict[f'spv_{batch_id_key}'] - spv_idx_subtract] * n_cells
        adata_dict['gene_panel_value'] = [self.token_dict[
            f'spv_{self.file_path_to_gene_panel_ID_dict[str(adata_file_path)]}']
            - spv_idx_subtract] * n_cells
        adata_dict['assay_value'] = [
            self.token_dict[
                f'spv_{adata.uns["assay"]}'] - spv_idx_subtract] * n_cells
        adata_dict['species_value'] = [self.token_dict[
            f'spv_{adata.uns["species"]}'] - spv_idx_subtract] * n_cells
        adata_dict['tissue_value'] = [self.token_dict[
            f'spv_{adata.uns["tissue"]}'] - spv_idx_subtract] * n_cells

        return adata_dict
            
    def _format_tokens_of_a_single_cell(self,
                                        example: dict
                                        ) -> dict:
        """
        Format tokens of a single cell.
        """
        # Get example-specific number of gene segments
        n_gene_segments = 1 # index cell segment
        if len(example['seg_tokens_neighborhood']) > 0:
            n_gene_segments += len(set(example['seg_tokens_neighborhood']))

        # Retrieve cell gene tokens and gene expression
        gene_tokens_cell, n_nonzero_cell_tokens = process_gene_tokens(
            example['gene_tokens_cell'],
            int(self.model_input_size / n_gene_segments),
            self.token_dict)
        del example['gene_tokens_cell']
        gene_expr_cell = process_gene_expr(
            example['gene_expr_cell'],
            int(self.model_input_size / n_gene_segments))
        del example['gene_expr_cell']

        # Retrieve neighborhood gene tokens and gene expression
        gene_tokens_neighborhood = np.array([])
        seg_tokens_neighborhood = np.array([])
        gene_expr_neighborhood = np.array([])
        n_nonzero_neighborhood_tokens = 0

        if n_gene_segments > 1:
            for segment in range(self.max_special_tokens + 1, n_gene_segments + self.max_special_tokens):
                gene_tokens_neighborhood_segment = [
                    example['gene_tokens_neighborhood'][i] for i in range(
                        len(example['gene_tokens_neighborhood']))
                        if example['seg_tokens_neighborhood'][i] == segment]
                seg_tokens_neighborhood_segment = [
                    example['seg_tokens_neighborhood'][i] for i in range(
                        len(example['seg_tokens_neighborhood']))
                        if example['seg_tokens_neighborhood'][i] == segment]

                gene_tokens_neighborhood_segment, n_nonzero_neighborhood_segment_tokens = process_gene_tokens(
                    gene_tokens_neighborhood_segment,
                    int(self.model_input_size / n_gene_segments),
                    self.token_dict)

                seg_tokens_neighborhood_segment, _ = process_gene_tokens(
                    seg_tokens_neighborhood_segment,
                    int(self.model_input_size / n_gene_segments),
                    self.token_dict)

                gene_tokens_neighborhood = np.hstack(
                    (gene_tokens_neighborhood, gene_tokens_neighborhood_segment))
                seg_tokens_neighborhood = np.hstack(
                    (seg_tokens_neighborhood, seg_tokens_neighborhood_segment))

                n_nonzero_neighborhood_tokens += n_nonzero_neighborhood_segment_tokens

                gene_expr_neighborhood_segment = [
                    example['gene_expr_neighborhood'][i] for i in range(
                        len(example['gene_expr_neighborhood']))
                        if example['seg_tokens_neighborhood'][i] == segment]

                gene_expr_neighborhood_segment = process_gene_expr(
                    gene_expr_neighborhood_segment,
                    int(self.model_input_size / n_gene_segments))

                gene_expr_neighborhood = np.hstack(
                    (gene_expr_neighborhood, gene_expr_neighborhood_segment))

        del example['gene_tokens_neighborhood']
        del example['gene_expr_neighborhood']
        del example['seg_tokens_neighborhood']
        example['gene_tokens'] = np.concatenate(
            (gene_tokens_cell.copy(), gene_tokens_neighborhood.copy())).astype(
                int)
        example['gene_expr'] = np.concatenate(
            (gene_expr_cell.copy(), gene_expr_neighborhood.copy())).astype(
                float)
        example['seg_tokens'] = np.concatenate(
            (np.array([self.max_special_tokens if gene_token != 0 else 0 for gene_token in
                       gene_tokens_cell]),
             seg_tokens_neighborhood.copy())).astype(int)

        # Retrieve attributes
        example['n_nonzero_tokens'] = (
            n_nonzero_cell_tokens + n_nonzero_neighborhood_tokens)

        # Add padding to make all sequences have length 'model_input_size'
        if len(example['gene_tokens']) < self.model_input_size:
            example['gene_tokens'] = np.append(
                example['gene_tokens'],
                np.zeros(self.model_input_size - len(example['gene_tokens']),
                         dtype=int))
            example['seg_tokens'] = np.append(
                example['seg_tokens'],
                np.zeros(self.model_input_size - len(example['seg_tokens']),
                         dtype=int))
            example['gene_expr'] = np.append(example['gene_expr'], np.zeros(
                (self.model_input_size - len(example['gene_expr']))))
        
        # Retrieve special tokens
        example['cls_tokens'] = [
            self.token_dict[f'<cls_{i}>'] for i in range(
                example['cell_degrees'] + 1)] # include cell itself
        example['cls_tokens'] += [0] * (
            self.max_cls_tokens - len(example['cls_tokens']))    

        example['assay_token'] = [example['assay_token']]
        example['species_token'] = [example['species_token']]
        example['tissue_token'] = [example['tissue_token']]
        example['gene_panel_token'] = [example['gene_panel_token']]
        example['batch_token'] = [example['batch_token']]

        example['assay_value_token'] = [example['assay_value_token']]
        example['species_value_token'] = [example['species_value_token']]
        example['tissue_value_token'] = [example['tissue_value_token']]
        example['gene_panel_value_token'] = [example['gene_panel_value_token']]
        example['batch_value_token'] = [example['batch_value_token']] 

        # Retrieve special token values
        example['assay_value'] = [example['assay_value']]
        example['species_value'] = [example['species_value']]
        example['tissue_value'] = [example['tissue_value']]
        example['gene_panel_value'] = [example['gene_panel_value']]
        example['batch_value'] = [example['batch_value']]

        return example


class CellNeighborhoodTokenizer(CellBaseTokenizer):
    def __init__(self,
                 **base_tokenizer_kwargs,
                 ):
        """
        CellNeighborhoodTokenizer class.

        Parameters
        -----------
        **base_tokenizer_kwargs:
            Keyword arguments for the initialization of the CellBaseTokenizer.
        """
        super().__init__(**base_tokenizer_kwargs)


    @staticmethod
    def process_chunk(chunk_start, chunk_size, adata, coding_miRNA_idx, 
                    coding_miRNA_tokens_cell, coding_miRNA_tokens_neighborhood, 
                    include_zero_expr_genes):
        chunk_data = {
            'gene_tokens_cell': [],
            'gene_tokens_neighborhood': [],
            'gene_expr_cell': [],
            'gene_expr_neighborhood': []
        }
        
        if include_zero_expr_genes:
            norm_counts_cell = adata[
                chunk_start : chunk_start + chunk_size, coding_miRNA_idx].X.toarray()
            norm_counts_neighborhood = adata[
                chunk_start : chunk_start + chunk_size, coding_miRNA_idx].layers[
                'X_neighborhood'].toarray()
            
            chunk_data['gene_tokens_cell'] = [
                rank_gene_tokens(norm_counts_cell[j], coding_miRNA_tokens_cell)
                for j in range(norm_counts_cell.shape[0])
            ]
            chunk_data['gene_tokens_neighborhood'] = [
                rank_gene_tokens(norm_counts_neighborhood[j], coding_miRNA_tokens_neighborhood)
                for j in range(norm_counts_neighborhood.shape[0])
            ]
            chunk_data['gene_expr_cell'] = [
                norm_counts_cell[j][np.argsort(-norm_counts_cell[j])]
                for j in range(norm_counts_cell.shape[0])
            ]
            chunk_data['gene_expr_neighborhood'] = [
                norm_counts_neighborhood[j][np.argsort(-norm_counts_neighborhood[j])]
                for j in range(norm_counts_neighborhood.shape[0])
            ]
        else:
            norm_counts_cell = sp.csr_matrix(adata[
                chunk_start : chunk_start + chunk_size, coding_miRNA_idx].X)
            norm_counts_neighborhood = sp.csr_matrix(adata[
                chunk_start : chunk_start + chunk_size, coding_miRNA_idx].layers[
                'X_neighborhood'])
            
            chunk_data['gene_tokens_cell'] = [
                rank_gene_tokens(norm_counts_cell[j].data, coding_miRNA_tokens_cell[norm_counts_cell[j].indices])
                for j in range(norm_counts_cell.shape[0])
            ]
            chunk_data['gene_tokens_neighborhood'] = [
                rank_gene_tokens(norm_counts_neighborhood[j].data, coding_miRNA_tokens_neighborhood[
                    norm_counts_neighborhood[j].indices])
                for j in range(norm_counts_neighborhood.shape[0])
            ]
            chunk_data['gene_expr_cell'] = [
                norm_counts_cell[j].data[np.argsort(-norm_counts_cell[j].data)]
                for j in range(norm_counts_cell.shape[0])
            ]
            chunk_data['gene_expr_neighborhood'] = [
                norm_counts_neighborhood[j].data[np.argsort(-norm_counts_neighborhood[j].data)]
                for j in range(norm_counts_neighborhood.shape[0])
            ]
        
        return chunk_data


    def _tokenize_adata(self,
                        adata_file_path: Path | str,
                        keep_in_memory: bool=False,
                        cache_directory_path: Optional[Path | str]=None,
                        ) -> dict:
        """
        Tokenize cells from an '.h5ad' (anndata) file.

        Parameters
        ----------
        adata_file_path:
            Path to anndata file containing cells to be tokenized.
        keep_in_memory:
            If 'True', keep dataset in memory when using generator.
        cache_directory_path:
            If specified, cache directory path for dataset creation.

        Returns
        ----------
        adata_dict:
            Dictionary with tokenized data stored in keys:
            - gene_tokens_cell:
                Cell-wise vector of ranked cell gene tokens.
            - gene_expr_cell:
                Cell-wise vector of ranked cell gene expression.
            - gene_tokens_neighborhood:
                Cell-wise vector of ranked neighborhood gene tokens.
            - gene_expr_neighborhood:
                Cell-wise vector of ranked neighborhood gene expression.
            - assay_token:
                List containing assay token.
            - species_tokens:
                List containing species token.
            - tissue_token:
                List containing tissue token.
            - gene_panel_token:
                List containing gene panel token.
            - batch_token:
                List containing batch token.
            - cell_ids:
                List of cell IDs.
        """
        # Initialize dict to collect tokens and cell ids
        adata_dict = {}

        adata = ad.read_h5ad(adata_file_path)

        print('Filtering cells.')
        # Filter to remove poor quality cells
        adata = filter_poor_quality_cells(adata)
        
        assert np.all(np.mod(adata.X.data, 1) == 0), "adata.X contains non-integer values."

        print('Computing spatial neighborhood graph and aggregating counts.')
        # Aggregate neighborhood cell gene expression
        adata = aggregate_neighbors(
            adata,
            n_neighs=self.n_neighs,
            radius=self.radius,
            delaunay=self.delaunay,
            include_self_loop=True)

        print('Normalizing gene expression counts.')
        # Perform normalization of total counts per cell
        if self.norm_factor == 'read_depth':
            if self.norm_method == 'analytic_pearson_residuals':
                raise ValueError(
                    "Invalid combination of 'norm_factor' and 'norm_method': "
                    f'({self.norm_factor} / {self.norm_method}).')
            adata.X = normalize_by_read_depth(adata.X)
            adata.layers['X_neighborhood'] = normalize_by_read_depth(
                adata.layers['X_neighborhood'])
        elif self.norm_factor == 'cell_area':
            if self.norm_method == 'analytic_pearson_residuals':
                raise ValueError(
                    "Invalid combination of 'norm_factor' and 'norm_method': "
                    f'({self.norm_factor} / {self.norm_method}).')
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
        coding_miRNA_ids = adata.var['ensembl_id'].iloc[coding_miRNA_idx]

        coding_miRNA_tokens_cell = np.array(
            [self.token_dict[gene_id] for gene_id in coding_miRNA_ids])
        coding_miRNA_tokens_neighborhood = np.array(
            [self.token_dict[gene_id] for gene_id in coding_miRNA_ids])

        # Prepare gene tokens for cell and neighborhood for this file
        adata_dict['gene_tokens_cell'] = []
        adata_dict['gene_expr_cell'] = []
        adata_dict['gene_tokens_neighborhood'] = []
        adata_dict['gene_expr_neighborhood'] = []

        num_threads = min(self.max_threads, int(len(adata)/self.chunk_size))

        # Divide cells into chunks and loop through chunks
        tracemalloc.start()
        if num_threads == 1:
            print(f"Ranking gene tokens based on normalized counts using 1 thread...")
            for i in range(0, len(adata), self.chunk_size):
                chunk_result = self.process_chunk(
                                    i,
                                    self.chunk_size,
                                    adata,
                                    coding_miRNA_idx,
                                    coding_miRNA_tokens_cell,
                                    coding_miRNA_tokens_neighborhood,
                                    self.include_zero_expr_genes
                                )
                adata_dict['gene_tokens_cell'].extend(chunk_result['gene_tokens_cell'])
                adata_dict['gene_tokens_neighborhood'].extend(chunk_result['gene_tokens_neighborhood'])
                adata_dict['gene_expr_cell'].extend(chunk_result['gene_expr_cell'])
                adata_dict['gene_expr_neighborhood'].extend(chunk_result['gene_expr_neighborhood'])
  
        elif num_threads > 1:
            print(f"Ranking gene tokens based on normalized counts using {num_threads} threads...")
            with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
                futures = {
                    executor.submit(
                        self.process_chunk, i, self.chunk_size, adata, coding_miRNA_idx, 
                        coding_miRNA_tokens_cell, coding_miRNA_tokens_neighborhood, 
                        self.include_zero_expr_genes
                    ): i
                    for i in range(0, len(adata), self.chunk_size)
                }
                
                results = []
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        chunk_index = futures[future]
                        chunk_result = future.result()  # Retrieve results
                        results.append((chunk_index, chunk_result))
                    except Exception as e:
                        print(f"Task failed with error: {e}")
            
            # This will sort the results in the order of the chunk index so that it matches with the cell_id order in the original adata.
            results.sort(key=lambda x: x[0])
            
            for _, chunk_result in results:
                adata_dict['gene_tokens_cell'].extend(chunk_result['gene_tokens_cell'])
                adata_dict['gene_tokens_neighborhood'].extend(chunk_result['gene_tokens_neighborhood'])
                adata_dict['gene_expr_cell'].extend(chunk_result['gene_expr_cell'])
                adata_dict['gene_expr_neighborhood'].extend(chunk_result['gene_expr_neighborhood'])

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Peak memory usage while Ranking Gene Tokens was {(peak / 10**6):.2f} MB.")
        print(f"Current memory usage is {(current / 10**6):.2f} MB.")

        # Add cell IDs for collecting metadata at inference time
        adata_dict['cell_id'] = adata.obs['cell_id'].values.tolist()
        
        # Add special tokens (positional tokens and value tokens)
        n_cells = len(adata)
        batch_id_key = f"{adata.uns['dataset_id']}_{adata.uns['batch']}"

        adata_dict['batch_token'] = [self.token_dict['spt_batch']] * n_cells
        adata_dict['gene_panel_token'] = [
            self.token_dict['spt_gene_panel']] * n_cells
        adata_dict['assay_token'] = [self.token_dict['spt_assay']] * n_cells
        adata_dict['species_token'] = [self.token_dict['spt_species']] * n_cells
        adata_dict['tissue_token'] = [self.token_dict['spt_tissue']] * n_cells

        adata_dict['batch_value_token'] = [
            self.token_dict[f'spv_{batch_id_key}']] * n_cells
        adata_dict['gene_panel_value_token'] = [self.token_dict[
            f'spv_{self.file_path_to_gene_panel_ID_dict[str(adata_file_path)]}']
            ] * n_cells
        adata_dict['assay_value_token'] = [
            self.token_dict[f'spv_{adata.uns["assay"]}']] * n_cells
        adata_dict['species_value_token'] = [
            self.token_dict[f'spv_{adata.uns["species"]}']] * n_cells
        adata_dict['tissue_value_token'] = [
            self.token_dict[f'spv_{adata.uns["tissue"]}']] * n_cells

        # Store values with right embedding index for count tokenizer
        # Leave space for <pad>, <mask> and <cls> tokens
        spv_dict = {
            k: v for k, v in self.token_dict.items() if k.startswith('spv_')}
        spv_start_idx = min(spv_dict.values())
        spv_idx_subtract = (spv_start_idx - 2 - self.max_cls_tokens)

        adata_dict['batch_value'] = [
            self.token_dict[f'spv_{batch_id_key}'] - spv_idx_subtract] * n_cells
        adata_dict['gene_panel_value'] = [self.token_dict[
            f'spv_{self.file_path_to_gene_panel_ID_dict[str(adata_file_path)]}']
            - spv_idx_subtract] * n_cells
        adata_dict['assay_value'] = [
            self.token_dict[
                f'spv_{adata.uns["assay"]}'] - spv_idx_subtract] * n_cells
        adata_dict['species_value'] = [self.token_dict[
            f'spv_{adata.uns["species"]}'] - spv_idx_subtract] * n_cells
        adata_dict['tissue_value'] = [self.token_dict[
            f'spv_{adata.uns["tissue"]}'] - spv_idx_subtract] * n_cells


        def example_generator():
            for i in range(n_cells):
                yield {k: adata_dict[k][i] for k in adata_dict.keys()}

        tracemalloc.start()
        dataset = Dataset.from_generator(
                                            example_generator,
                                            num_proc=num_threads,
                                            keep_in_memory=keep_in_memory,
                                            cache_dir=cache_directory_path,
                                            writer_batch_size=self.chunk_size,
                                        )
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Peak memory usage while Generating Batch Dataset with {n_cells} cells was {(peak / 10**6):.2f} MB.")
        print(f"Current memory usage is {(current / 10**6):.2f} MB.")
        print(f"Completed Tokenizing {adata_file_path}.")
        
        return dataset


    def _format_tokens_of_a_single_cell(self,
                                        example: dict
                                        ) -> dict:
        """
        Format tokens of a single cell.
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
            (gene_tokens_cell.copy(), gene_tokens_neighborhood.copy())).astype(
                int)

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
            (gene_expr_cell.copy(), gene_expr_neighborhood.copy())).astype(
                float)

        # Retrieve attributes
        example['n_nonzero_tokens'] = (
            n_nonzero_cell_tokens + n_nonzero_neighborhood_tokens)

        # Define segments (leave space for special token segments)
        example['seg_tokens'] = np.concatenate(
            (np.array([self.max_special_tokens if gene_token != 0 else 0 for 
                       gene_token in gene_tokens_cell]),
             np.array([self.max_special_tokens + 1 if gene_token != 0 else 0
                       for gene_token in gene_tokens_neighborhood])
                       )).astype(int)
        
        # Retrieve special tokens
        example['cls_tokens'] = [self.token_dict['<cls_cell>'],
                                 self.token_dict['<cls_neighborhood>']]
        example['assay_token'] = [example['assay_token']]
        example['species_token'] = [example['species_token']]
        example['tissue_token'] = [example['tissue_token']]
        example['gene_panel_token'] = [example['gene_panel_token']]
        example['batch_token'] = [example['batch_token']]

        example['assay_value_token'] = [example['assay_value_token']]
        example['species_value_token'] = [example['species_value_token']]
        example['tissue_value_token'] = [example['tissue_value_token']]
        example['gene_panel_value_token'] = [example['gene_panel_value_token']]
        example['batch_value_token'] = [example['batch_value_token']]        

        # Retrieve special token values
        example['assay_value'] = [example['assay_value']]
        example['species_value'] = [example['species_value']]
        example['tissue_value'] = [example['tissue_value']]
        example['gene_panel_value'] = [example['gene_panel_value']]
        example['batch_value'] = [example['batch_value']]

        return example