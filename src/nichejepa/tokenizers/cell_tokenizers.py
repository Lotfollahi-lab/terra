"""
Adapted from Theodoris, C. V. et al. Transfer learning enables
predictions in network biology. Nature 618, 616–624 (2023);
https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/tokenizer.py
(12.04.2024).

Input Data
----------
Required format:
    Raw counts spatial transcriptomics (ST) data with all genes (no
    feature selection) as '.h5ad' (AnnData) files. Spatial coordinates
    are stored in adata.obsm['spatial'].
Required gene attributes:
    Ensembl ID for each gene (adata.var['ensembl_id']).
Required cell attributes:
    Cell ID in index. Metadata is retrieved at inference time via this
    cell ID.
Optional cell attributes:
    Binary indicator of whether cell should be included in tokenization
    based on user-defined filtering criteria (adata.obs['filter_pass']).

Usage
----------
.. code-block :: python
    >>> from nichejepa import CellGraphTokenizer
    >>> tk = CellGraphTokenizer(nproc=4)
    >>> tk.tokenize_data(
    >>>     'input_directory', 'output_directory', 'output_file_prefix')

or

.. code-block :: python
    >>> from nichejepa import CellNeighborhoodTokenizer
    >>> tk = CellNeighborhoodTokenizer(nproc=4)
    >>> tk.tokenize_data(
    >>>     'input_directory', 'output_directory', 'output_file_prefix')

Description
----------
Input data is a directory with '.h5ad' files containing raw counts from
ST data, including all genes detected without feature selection. The
input file type is specified by the argument `file_format` in the
`tokenize_data()` function. Genes should be labeled with Ensembl IDs
(adata.var['ensembl_id']), which provide a unique identifer for
conversion to tokens. Gene names can be converted to Ensembl IDs via the
helper function `nichejepa.datasets.utils.get_ensembl_ids()` or via the
pyensembl Python package. No cell metadata is required, but the cell ID
needs to be stored in the index. Additionally, if the original '.h5ad'
file contains a cell attribute called adata.obs['filter_pass'], this can
be used as a binary indicator of whether to include these cells in the
tokenization. All cells with '1' in this attribute will be tokenized,
whereas the others will be excluded. One may use this column to indicate
QC filtering or other criteria for selection for inclusion in the final
tokenized dataset. If one's data is in other formats besides '.h5ad',
one should the relevant tools (such as AnnData tools) to convert the
file to '.h5ad' format prior to initializing the cell tokenizer.
"""

from __future__ import annotations

import concurrent
import logging
import pickle
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import squidpy as sq
from datasets import Dataset

from ..preprocessors.filters import filter_cells
from ..preprocessors.graph import construct_neighbor_graph
from ..preprocessors.normalizers import normalize_by_analytic_pearson_residuals
from ..preprocessors.normalizers import normalize_by_cell_area
from ..preprocessors.normalizers import normalize_by_gene_corrected_read_depth
from ..preprocessors.normalizers import normalize_by_factor
from ..preprocessors.normalizers import normalize_by_read_depth
from ..preprocessors.normalizers import normalize_by_seurat
from ..preprocessors.normalizers import normalize_by_shifted_log
from .tokenize import process_gene_expr, process_gene_tokens, rank_gene_tokens


warnings.filterwarnings('ignore', message=".*The 'nopython' keyword.*") # noqa
logger = logging.getLogger(__name__)


base_path = Path(__file__).parent.parent.parent.parent
norm_factor_file_path = base_path / 'norm_factors.csv'
token_dictionary_file_path = base_path / 'token_dictionary.pkl'


class CellBaseTokenizer(ABC):
    """
    CellBaseTokenizer class.

    Parameters
    ----------
    n_proc:
        Number of processes.
    processing_mode:
        Processing mode.
    chunk_size:
        Chunk size used for splitting adata objects during tokenization.
    model_input_size:
        Sequence length of the cell sequence.
    include_zero_expr_genes:
        If `True`, include non-expressed genes in the tokenization.
    n_neighs:
        If specified, use `n_neighs` to compute the neighborhood graph.
        If `radius` or `delaunay` are also specified, a union
        neighborhood graph will be computed.
    radius:
        If specified, use `radius` to compute the neighborhood graph. If
        `n_neighs` or `delaunay` are also specified, a union
        neighborhood graph will be computed.
    delaunay:
        If `True`, compute the neighborhood graph by delaunay
        triangulation. If 'n_neighs' or 'radius' are also specified, a
        union neighborhooh graph will be computed.
    rank_cell_norm_method:
        Normalization method on cell level for ranking genes.
    rank_gene_norm_method:
        Normalization method on gene level for ranking genes.
    rank_count_norm_method:
        Normalization method on count level for ranking genes.
    count_cell_norm_method:
        Normalization method on cell level for gene expression.
    count_gene_norm_method:
        Normalization method on gene level for gene expression.
    count_count_norm_method:
        Normalization method on count level for gene expression.
    norm_factor_file_path:
        File path to '.csv' file containing norm factors per gene.
    token_dictionary_file_path:
        File path to the '.pkl' file containing the token dictionary.
    """
    def __init__(
            self,
            nproc: int = 1,
            processing_mode: Literal['parallel', 'sequential'] = 'sequential',
            chunk_size: int = 512,
            model_input_size: int = 2048,
            include_zero_expr_genes: bool = False,
            n_neighs: float | None = 10,
            radius: float | None = None,
            delaunay: bool = False,
            rank_cell_norm_method: Literal[
                'read_depth',
                'gene_corrected_read_depth',
                'cell_area',
                ] | None = 'gene_corrected_read_depth',
            rank_gene_norm_method: Literal[
                'mean',
                'nonzero_mean',
                'seurat_v3',
                ] | None = 'nonzero_mean',
            rank_count_norm_method: Literal[
                'analytic_pearson_residuals',
                'shifted_log',
                ] | None = None,
            count_cell_norm_method: Literal[
                'read_depth',
                'gene_corrected_read_depth',
                'cell_area',
                ] | None = None,
            count_gene_norm_method: Literal[
                'mean',
                'nonzero_mean',
                'seurat_v3',
                ] | None = None,
            count_count_norm_method: Literal[
                'analytic_pearson_residuals',
                'shifted_log',
                ] | None = 'shifted_log',
            norm_factor_file_path: Path | str = norm_factor_file_path,
            token_dictionary_file_path: Path | str = token_dictionary_file_path,
            ):
        self.nproc = nproc
        self.processing_mode = processing_mode
        self.chunk_size = chunk_size
        self.model_input_size = model_input_size
        self.include_zero_expr_genes = include_zero_expr_genes
        self.n_neighs = n_neighs
        self.radius = radius
        self.delaunay = delaunay
        self.rank_cell_norm_method = rank_cell_norm_method
        self.rank_gene_norm_method = rank_gene_norm_method
        self.rank_count_norm_method = rank_count_norm_method
        self.count_cell_norm_method = count_cell_norm_method
        self.count_gene_norm_method = count_gene_norm_method
        self.count_count_norm_method = count_count_norm_method
        self.norm_factor_file_path = norm_factor_file_path
        self.token_dictionary_file_path = token_dictionary_file_path

        # TODO fix this for radius and delaunay
        self.seq_len_cell = int(self.model_input_size / (self.n_neighs + 1))

        # Load token dictionary
        logger.info('Loading token dictionary from '
                    f'{self.token_dictionary_file_path}.')
        with open(token_dictionary_file_path, 'rb') as f:
            self.token_dict = pickle.load(f)

        # Get maximum number of cls and special tokens based on token
        # dict
        self.max_cls_tokens = sum(1 for key in self.token_dict if "cls" in key)
        self.max_special_tokens = self.max_cls_tokens + sum(
            1 for key in self.token_dict if "spt" in key)
            
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
                      file_format: Literal['h5ad'] = 'h5ad',
                      use_generator: bool = False,
                      cache_directory_path: Path | str | None = None,
                      num_shards: int | None = None,
                      keep_in_memory: bool = False,
                      ):
        """
        Tokenize files in `input_directory` and save as tokenized
        `.dataset` file in `output_directory`.

        Parameters
        ----------
        input_directory:
            Path to directory containing `.h5ad` (AnnData) files.
        output_directory:
            Path to directory where tokenized data will be saved as `.dataset`
            file.
        output_file_prefix:
            Prefix for output file.
        file_format:
            Format of input files. Must be `.h5ad` currently.
        use_generator:
            If `True`, use generator for tokenization, else dict.
        cache_directory_path:
            If specified, cache directory path for dataset creation.
        num_shards:
            Number of shards to save dataset to.
        keep_in_memory:
            If `True`, keep dataset in memory when using generator.
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
                        use_generator: bool = False,
                        cache_directory_path: Path | str | None = None,
                        keep_in_memory: bool = False,
                        ) -> Dataset:
        """
        Create a Hugging Face dataset based on tokenized cells.

        Parameters
        ----------
        dataset_dict:
            Dictionary based on which the Hugging Face dataset will be
            created.
        use_generator:
            If 'True', use generator for tokenization, else dict.
        cache_directory_path:
            If specified, cache directory path for dataset creation.
        keep_in_memory:
            If 'True', keep dataset in memory when using generator.

        Returns
        ----------
        dataset:
            Hugging Face dataset containing the tokenized cells.
        """
        # Create Hugging Face dataset
        logger.info('Creating Hugging Face dataset...')
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
                        file_format: Literal['h5ad'] = 'h5ad',
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
            Dictionary containing the cell IDs and tokens for the
            tokenized files.
        """
        file_found = 0

        tokenize_file_fn = self._tokenize_adata

        # Initialize dict to add results from individual files
        dataset_dict = {}

        # Loop through data directory to tokenize `.h5ad`` files
        if self.processing_mode == 'sequential':
            logger.info('Tokenizing files sequentially...')
            for file_path in data_directory.glob(f'**/*.{file_format}'):
                file_found = 1
                logger.info(f"Tokenizing '{file_path}'...")
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
                for file_path in data_directory.glob(f'**/*.{file_format}'):
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
    def __init__(self,
                 **base_tokenizer_kwargs,
                 ):
        """
        CellGraphTokenizer class.

        Parameters
        -----------
        **base_tokenizer_kwargs:
            Keyword arguments for the initialization of the
            CellBaseTokenizer.
        """
        super().__init__(**base_tokenizer_kwargs)

    def _tokenize_adata(self,
                        adata_file_path: Path | str | None = None,
                        adata: ad.AnnData | None = None,
                        ) -> dict:
        """
        Tokenize cells from an `.h5ad` (AnnData) file, equivalent to one
        batch.

        Parameters
        ----------
        adata_file_path:
            Path to AnnData file containing cells to be tokenized.
        adata:
            AnnData object to be tokenized.

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
                Segment tokens for the neighborhood (each neighbor cell
                is a different segment).
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
            - cell_total_counts:
                Cell and neighbor cell read depth.
            - cell_n_probed_genes:
                Number of genes probed.
        """
        # Initialize dict to collect tokens, cell ids, and metadata
        adata_dict = {}

        # Read batch
        if adata is None:
            if adata_file_path is not None:
                adata = ad.read_h5ad(adata_file_path)
            else:
                raise ValueError(
                    'Specify either `adata` or `adata_file_path`.')
        else:
            if adata_file_path is not None:
                raise ValueError(
                    'Specify either `adata` or `adata_file_path`, not both.')

        print('Filtering cells...')
        # Filter cells based on adata.obs['filter_pass']
        adata = filter_cells(adata)

        adata_neigh = adata        

        print('Computing spatial neighborhood...')
        # Construct neighbor graph
        adata_neigh = construct_neighbor_graph(
            adata_neigh,
            n_neighs=self.n_neighs,
            radius=self.radius,
            delaunay=self.delaunay,
            include_self_loop=False,
            compute_neighbor_counts=False)

        print('Normalizing gene expression counts...')
        # Perform normalization of counts per cell for rank and count
        # tokenization
        if self.rank_cell_norm_method == 'read_depth':
            adata.layers['X_rank'] = normalize_by_read_depth(adata.X)

        elif self.rank_cell_norm_method == 'gene_corrected_read_depth':
            adata.layers['X_rank'] = \
                normalize_by_gene_corrected_read_depth(adata.X)

        elif self.rank_cell_norm_method == 'cell_area':
            adata.layers['X_rank'] = normalize_by_cell_area(
                adata.X,
                cell_areas=adata.obs['cell_area'].values)
        else:
            if self.rank_cell_norm_method is None:
                adata.layers['X_rank'] = adata.X
            else:
                raise ValueError(
                    f"Invalid 'cell_norm_method' {self.rank_cell_norm_method}.")

        if self.count_cell_norm_method == 'read_depth':
            adata.layers['X_count'] = normalize_by_read_depth(adata.X)                

        elif self.count_cell_norm_method == 'gene_corrected_read_depth':
            adata.layers['X_count'] = \
                normalize_by_gene_corrected_read_depth(adata.X)                   

        elif self.count_cell_norm_method == 'cell_area':
            adata.layers['X_count'] = normalize_by_cell_area(
                adata.X,
                cell_areas=adata.obs['cell_area'].values)
        else:
            if self.count_cell_norm_method is None:
                adata.layers['X_count'] = adata.X
            else:
                raise ValueError(
                    f"Invalid 'cell_norm_method' \
                    {self.count_cell_norm_method}.")

        # Perform normalization of counts per gene for rank and count
        # tokenization
        if self.rank_gene_norm_method == 'mean':
            if self.rank_cell_norm_method is None:
                norm_factor = 'mean'
            elif self.rank_cell_norm_method == 'read_depth':
                norm_factor = 'read_depth_mean'
            elif self.rank_cell_norm_method == 'gene_corrected_read_depth':
                norm_factor = 'gene_corrected_read_depth_mean'
            elif self.rank_cell_norm_method == 'cell_area':
                norm_factor = 'cell_area_mean'
            adata.layers['X_rank'] = normalize_by_factor(
                adata.layers['X_rank'],
                norm_factor_file_path=self.norm_factor_file_path,
                probed_genes=adata.var['ensembl_id'],
                norm_factor=norm_factor)
        elif self.rank_gene_norm_method == 'nonzero_mean':
            if self.rank_cell_norm_method is None:
                norm_factor = 'nonzero_mean'
            elif self.rank_cell_norm_method == 'read_depth':
                norm_factor = 'read_depth_nonzero_mean'
            elif self.rank_cell_norm_method == 'gene_corrected_read_depth':
                norm_factor = 'gene_corrected_read_depth_nonzero_mean'
            elif self.rank_cell_norm_method == 'cell_area':
                norm_factor = 'cell_area_nonzero_mean'
            adata.layers['X_rank'] = normalize_by_factor(
                adata.layers['X_rank'],
                norm_factor_file_path=self.norm_factor_file_path,
                probed_genes=adata.var['ensembl_id'],
                norm_factor=norm_factor)              
        elif self.rank_gene_norm_method == 'seurat_v3':
            adata.layers['X_rank'] = normalize_by_seurat(adata.X)
        else:
            if self.rank_gene_norm_method is None:
                pass
            else:
                raise ValueError(
                    f"Invalid 'gene_norm_method' {self.rank_gene_norm_method}.")

        if self.count_gene_norm_method == 'mean':
            if self.count_cell_norm_method is None:
                norm_factor = 'mean'
            elif self.count_cell_norm_method == 'read_depth':
                norm_factor = 'read_depth_mean'
            elif self.count_cell_norm_method == 'gene_corrected_read_depth':
                norm_factor = 'gene_corrected_read_depth_mean'
            elif self.count_cell_norm_method == 'cell_area':
                norm_factor = 'cell_area_mean'
            adata.layers['X_count'] = normalize_by_factor(
                adata.layers['X_count'],
                norm_factor_file_path=self.norm_factor_file_path,
                probed_genes=adata.var['ensembl_id'],
                norm_factor=norm_factor)              
        elif self.count_gene_norm_method == 'nonzero_mean':
            if self.count_cell_norm_method is None:
                norm_factor = 'nonzero_mean'
            elif self.count_cell_norm_method == 'read_depth':
                norm_factor = 'read_depth_nonzero_mean'
            elif self.count_cell_norm_method == 'gene_corrected_read_depth':
                norm_factor = 'gene_corrected_read_depth_nonzero_mean'
            elif self.count_cell_norm_method == 'cell_area':
                norm_factor = 'cell_area_nonzero_mean'
            adata.layers['X_count'] = normalize_by_factor(
                adata.layers['X_count'],
                norm_factor_file_path=self.norm_factor_file_path,
                probed_genes=adata.var['ensembl_id'],
                norm_factor=norm_factor)            
        elif self.count_gene_norm_method == 'seurat_v3':
            adata.layers['X_count'] = normalize_by_seurat(adata.X)
        else:
            if self.count_gene_norm_method is None:
                pass
            else:
                raise ValueError(
                    f"Invalid 'gene_norm_method' \
                    {self.count_gene_norm_method}.")

        # Perform normalization of counts for rank and count tokenization
        if self.rank_count_norm_method == 'analytic_pearson_residuals':
            if (self.rank_cell_norm_method is not None) or (
                self.rank_gene_norm_method is not None):
                raise ValueError('Invalid combination of norm methods.')
            adata.layers['X_rank'] = \
                normalize_by_analytic_pearson_residuals(adata.layers['X_rank'])           
        elif self.rank_count_norm_method == 'shifted_log':
            adata.layers['X_rank'] = normalize_by_shifted_log(
                adata.layers['X_rank'])               
        else:
            if self.rank_count_norm_method is None:
                pass
            else:
                raise ValueError(
                    f"Invalid 'counts_norm_method': \
                    {self.rank_count_norm_method}.")

        if self.count_count_norm_method == 'analytic_pearson_residuals':
            if (self.count_cell_norm_method is not None) or (
                self.count_gene_norm_method is not None):
                raise ValueError('Invalid combination of norm methods.')
            adata.layers['X_count'] = \
                normalize_by_analytic_pearson_residuals(adata.layers['X_count'])            
        elif self.count_count_norm_method == 'shifted_log':
            adata.layers['X_count'] = normalize_by_shifted_log(
                adata.layers['X_count'])               
        else:
            if self.count_count_norm_method is None:
                pass
            else:
                raise ValueError(
                    f"Invalid 'counts_norm_method': \
                    {self.count_count_norm_method}.")

        # Initialize dict to collect tokens and cell IDs
        adata_dict = {}

        # Retrieve gene tokens for genes contained in batch and vocab, i.e.
        # protein-coding and miRNA genes
        print('Retrieving gene tokens.')
        coding_miRNA_idx = np.where(
            [self.coding_miRNA_dict.get(
                gene_id, False) for gene_id in adata.var['ensembl_id']])[0]
        coding_miRNA_ids = adata.var['ensembl_id'][coding_miRNA_idx]

        coding_miRNA_tokens_cell = np.array(
            [self.token_dict[gene_id] for gene_id in coding_miRNA_ids])

        # Add coordinate tokens of index cells
        adata_dict['rel_x_coord'] = [
            [0] * self.seq_len_cell for coord in adata.obsm['spatial'][:, 0].tolist()]
        adata_dict['rel_y_coord'] = [
            [0] * self.seq_len_cell for coord in adata.obsm['spatial'][:, 1].tolist()]

        # Add coordinate tokens of neighbor cells
        for i in range(len(adata)):
            neighbor_indices = adata_neigh.obsp[
                'spatial_connectivities'][i].indices
            for j in neighbor_indices:
                adata_dict['rel_x_coord'][i].extend([(
                    adata.obsm['spatial'][j, 0] - adata.obsm['spatial'][i, 0]
                    ).tolist()] * self.seq_len_cell)
                adata_dict['rel_y_coord'][i].extend([(
                    adata.obsm['spatial'][j, 1] - adata.obsm['spatial'][i, 1]
                    ).tolist()] * self.seq_len_cell)

        # Prepare gene tokens for cell and neighborhood for this batch
        adata_dict['gene_tokens_cell'] = []
        adata_dict['gene_expr_cell'] = []
        adata_dict['gene_tokens_neighborhood'] = []
        adata_dict['gene_expr_neighborhood'] = []
            
        # Divide cells into chunks and loop through chunks
        print('Ranking gene tokens based on normalized counts.')
        for i in range(0, len(adata), self.chunk_size):
            if self.include_zero_expr_genes:
                norm_counts_cell_rank = adata[
                    i : i + self.chunk_size, coding_miRNA_idx].layers[
                        'X_rank'].toarray()
                norm_counts_cell_count = adata[
                    i : i + self.chunk_size, coding_miRNA_idx].layers[
                        'X_count'].toarray()

                # Rank gene tokens and append across chunks
                adata_dict['gene_tokens_cell'] += [
                    rank_gene_tokens(norm_counts_cell_rank[j],
                    coding_miRNA_tokens_cell)
                    for j in range(norm_counts_cell_rank.shape[0])]

                # Rank gene expression and append across chunks
                adata_dict['gene_expr_cell'] += [
                    norm_counts_cell_count[j][
                        np.argsort(-norm_counts_cell_rank[j])]
                    for j in range(norm_counts_cell_count.shape[0])]

            else:
                norm_counts_cell_rank = sp.csr_matrix(adata[
                    i : i + self.chunk_size, coding_miRNA_idx].layers[
                        'X_rank'])
                norm_counts_cell_count = sp.csr_matrix(adata[
                    i : i + self.chunk_size, coding_miRNA_idx].layers[
                        'X_count'])              

                # Rank gene tokens and append across chunks
                adata_dict['gene_tokens_cell'] += [
                    rank_gene_tokens(
                        norm_counts_cell_rank[j].data,
                        coding_miRNA_tokens_cell[norm_counts_cell_rank[j].indices])
                    for j in range(norm_counts_cell_rank.shape[0])]

                # Rank gene expression and append across chunks
                adata_dict['gene_expr_cell'] += [
                    norm_counts_cell_count[j].data[
                        np.argsort(-norm_counts_cell_rank[j].data)]
                    for j in range(norm_counts_cell_count.shape[0])]

        adata_dict['gene_tokens_cell_neigh'] = adata_dict[
            'gene_tokens_cell']
        adata_dict['gene_expr_cell_neigh'] = adata_dict[
            'gene_expr_cell']

        print('Retrieving tokens for neighborhood cells.')
        adata_dict['gene_tokens_neighborhood'] = [
            np.array([]) for i in range(len(adata))]
        adata_dict['gene_expr_neighborhood'] = [
            np.array([]) for i in range(len(adata))]
        adata_dict['seg_tokens_neighborhood'] = [
            np.array([]) for i in range(len(adata))]
        
        #adata_dict['cell_degrees'] = []

        # Add cell IDs for cell identification when applying perturbations
        adata_dict['cell_ids'] = [
            [cell_id] for cell_id in adata.obs['cell_id'].values.tolist()]

        n_cells = len(adata)

        # Add read depth and number of genes
        #adata_dict['cell_total_counts'] = [
        #    [total_counts] for total_counts in adata.X.sum(axis=1).A1.tolist()]
        #adata_dict['cell_n_probed_genes'] = [adata.X.shape[1]] * n_cells
        
        # Loop through all cells to add neighbor cell gene tokens based on
        # position of neighbor cell compared to index cell. Gene tokens of cells
        # that are closer to the index cell will be added first.
        for i in range(len(adata_neigh)):
            # Collect all neighbors of cell i
            neighbors_i = adata_neigh.obsp[
                'spatial_connectivities'][i].nonzero()[1]

            # Store cell degree (number of neighbors excluding cell i)
            #adata_dict['cell_degrees'].append(int(len(neighbors_i)))

            # Take into account case where neighbor cells can have 0 distance
            if (adata_neigh.obsp['spatial_connectivities'].getnnz(axis=1)[i] != 
            adata_neigh.obsp['spatial_distances'].getnnz(axis=1)[i]):
                cell_con_nz = np.nonzero(
                    adata_neigh.obsp['spatial_connectivities'][i])[1]
                cell_dist_nz = np.nonzero(
                    adata_neigh.obsp['spatial_distances'][i])[1]
                zero_dist_idx = set(cell_con_nz) - set(cell_dist_nz)
                for idx in zero_dist_idx:
                    adata_neigh.obsp[
                        'spatial_distances'][i, idx] = adata_neigh.obsp[
                        'spatial_distances'][i, idx] + 10**(-9)
            
            # Get sorted indices of neighbor cells based on (lower) distance to
            # index cell
            cell_start = adata_neigh.obsp['spatial_distances'].indptr[i]
            cell_end = adata_neigh.obsp['spatial_distances'].indptr[i+1]
            cell_distances = adata_neigh.obsp[
                'spatial_distances'].data[cell_start:cell_end]

            sorted_indices = np.argsort(cell_distances)
            assert len(neighbors_i) == len(cell_distances), (
                'Number of neighbors does not equal number of distances.')

            # Loop through distance-sorted neighbor cells and add gene and
            # segment tokens and counts
            for j, k in enumerate(neighbors_i[sorted_indices]):
                adata_dict['gene_tokens_neighborhood'][i] = np.hstack(
                    (adata_dict['gene_tokens_neighborhood'][i],
                     adata_dict['gene_tokens_cell_neigh'][k]))
                adata_dict['gene_expr_neighborhood'][i] = np.hstack(
                    (adata_dict['gene_expr_neighborhood'][i],
                     adata_dict['gene_expr_cell_neigh'][k]))
                adata_dict['seg_tokens_neighborhood'][i] = np.hstack(
                    (adata_dict['seg_tokens_neighborhood'][i],
                     [j + self.max_special_tokens + 1] * len(
                        adata_dict['gene_tokens_cell_neigh'][k])))

                #adata_dict['cell_total_counts'][i].append(adata.X[k].sum())
                adata_dict['cell_ids'][i].extend(
                        [adata.obs['cell_id'].values.tolist()[k]] * len(
                        adata_dict['gene_tokens_cell_neigh'][k]))

        del adata_dict['gene_tokens_cell_neigh']
        del adata_dict['gene_expr_cell_neigh']

        # Add cell IDs for collecting metadata at inference time
        adata_dict['cell_id'] = adata.obs['cell_id'].values.tolist()
        
        # Add special tokens (positional tokens and value tokens)
        batch_id_key = f"{adata.uns['dataset_id']}_{adata.uns['batch']}"

        adata_dict['batch_token'] = [self.token_dict['spt_batch']] * n_cells
        adata_dict['gene_panel_token'] = [
            self.token_dict['spt_gene_panel']] * n_cells
        adata_dict['assay_token'] = [self.token_dict['spt_assay']] * n_cells
        adata_dict['species_token'] = [self.token_dict['spt_species']] * n_cells
        adata_dict['tissue_token'] = [self.token_dict['spt_tissue']] * n_cells

        #adata_dict['batch_value_token'] = [
        #    self.token_dict[f'spv_{batch_id_key}']] * n_cells
        #adata_dict['gene_panel_value_token'] = [self.token_dict[
        #    f'spv_gene_panel{len(adata.var_names)}']
        #    ] * n_cells
        #adata_dict['assay_value_token'] = [
        #    self.token_dict[f'spv_{adata.uns["assay"]}']] * n_cells
        #adata_dict['species_value_token'] = [
        #    self.token_dict[f'spv_{adata.uns["species"]}']] * n_cells
        #adata_dict['tissue_value_token'] = [
        #    self.token_dict[f'spv_{adata.uns["tissue"]}']] * n_cells

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
            f'spv_gene_panel{len(adata.var_names)}'] - spv_idx_subtract
            ] * n_cells
        adata_dict['assay_value'] = [
            self.token_dict[
                f'spv_{adata.uns["assay"]}'] - spv_idx_subtract] * n_cells
        adata_dict['species_value'] = [self.token_dict[
            f'spv_{adata.uns["species"]}'] - spv_idx_subtract] * n_cells
        adata_dict['tissue_value'] = [self.token_dict[
            f'spv_{adata.uns["tissue"]}'] - spv_idx_subtract] * n_cells

        return adata_dict
            
    def _format_examples(self,
                         example: dict) -> dict:
        """
        Format examples.
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
            for segment in range(self.max_special_tokens + 1, self.max_special_tokens + n_gene_segments):
                gene_tokens_neighborhood_segment = [
                    gene for gene, seg in zip(
                        example['gene_tokens_neighborhood'],
                        example['seg_tokens_neighborhood']) 
                    if seg == segment]
                seg_tokens_neighborhood_segment = [
                    seg for seg in example['seg_tokens_neighborhood'] 
                    if seg == segment]

                gene_tokens_neighborhood_segment, \
                n_nonzero_neighborhood_segment_tokens = process_gene_tokens(
                    gene_tokens_neighborhood_segment,
                    int(self.model_input_size / n_gene_segments),
                    self.token_dict)

                seg_tokens_neighborhood_segment, _ = process_gene_tokens(
                    seg_tokens_neighborhood_segment,
                    int(self.model_input_size / n_gene_segments),
                    self.token_dict)

                gene_tokens_neighborhood = np.hstack(
                    (gene_tokens_neighborhood,
                     gene_tokens_neighborhood_segment))
                seg_tokens_neighborhood = np.hstack(
                    (seg_tokens_neighborhood, seg_tokens_neighborhood_segment))

                n_nonzero_neighborhood_tokens += n_nonzero_neighborhood_segment_tokens

                gene_expr_neighborhood_segment = [
                    gene_expr for gene_expr, seg in zip(
                        example['gene_expr_neighborhood'],
                        example['seg_tokens_neighborhood']) 
                    if seg == segment]

                gene_expr_neighborhood_segment = process_gene_expr(
                    gene_expr_neighborhood_segment,
                    int(self.model_input_size / n_gene_segments))

                gene_expr_neighborhood = np.hstack(
                    (gene_expr_neighborhood, gene_expr_neighborhood_segment))

        del example['gene_tokens_neighborhood']
        del example['gene_expr_neighborhood']
        del example['seg_tokens_neighborhood']
        example['gene_tokens'] = np.concatenate(
            (gene_tokens_cell, gene_tokens_neighborhood)).astype(int)
        example['gene_expr'] = np.concatenate(
            (gene_expr_cell, gene_expr_neighborhood)).astype(float)
        example['seg_tokens'] = np.concatenate(
            (np.array([self.max_special_tokens if gene_token != 0 else 0
                       for gene_token in gene_tokens_cell]),
             seg_tokens_neighborhood)).astype(int)

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
        #example['cls_tokens'] = [
        #    self.token_dict[f'<cls_{i}>'] for i in range(
        #        example['cell_degrees'] + 1)] # include cell itself
        #example['cls_tokens'] += [0] * (
        #    self.max_cls_tokens - len(example['cls_tokens']))    

        example['assay_token'] = [example['assay_token']]
        example['species_token'] = [example['species_token']]
        example['tissue_token'] = [example['tissue_token']]
        example['gene_panel_token'] = [example['gene_panel_token']]
        example['batch_token'] = [example['batch_token']]

        #example['assay_value_token'] = [example['assay_value_token']]
        #example['species_value_token'] = [example['species_value_token']]
        #example['tissue_value_token'] = [example['tissue_value_token']]
        #example['gene_panel_value_token'] = [example['gene_panel_value_token']]
        #example['batch_value_token'] = [example['batch_value_token']] 

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
            Keyword arguments for the initialization of the
            CellBaseTokenizer.
        """
        super().__init__(**base_tokenizer_kwargs)

    def _tokenize_adata(self,
                        adata_file_path: Path | str | None = None,
                        adata: ad.AnnData | None = None,
                        ) -> dict:
        """
        Tokenize cells from an `.h5ad` (anndata) file.

        Parameters
        ----------
        adata_file_path:
            Path to anndata file containing cells to be tokenized.

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
            - cell_total_counts:
                Cell and neighborhood read depth.
            - cell_n_probed_genes:
                Number of genes probed.
        """
        # Initialize dict to collect tokens and cell ids
        adata_dict = {}

        if adata is None:
            adata = ad.read_h5ad(adata_file_path)

        print('Filtering cells.')
        # Filter to remove poor quality cells
        adata = filter_cells(adata)

        print('Computing spatial neighborhood graph and aggregating counts.')
        # Aggregate neighborhood cell gene expression
        adata = construct_neighbor_graph(
            adata,
            n_neighs=self.n_neighs,
            radius=self.radius,
            delaunay=self.delaunay,
            include_self_loop=True,
            compute_neighbor_counts=True)

        print('Normalizing gene expression counts...')
        # Perform normalization of counts per cell for rank and count
        # tokenization
        if self.rank_cell_norm_method == 'read_depth':
            adata.layers['X_rank'] = normalize_by_read_depth(adata.X)
            adata.layers['X_neighborhood_rank'] = normalize_by_read_depth(
                adata.layers['X_neighborhood'])

        elif self.rank_cell_norm_method == 'gene_corrected_read_depth':
            adata.layers['X_rank'] = normalize_by_gene_corrected_read_depth(
                adata.X)
            adata.layers['X_neighborhood_rank'] = normalize_by_gene_corrected_read_depth(
                adata.layers['X_neighborhood']) 

        elif self.rank_cell_norm_method == 'cell_area':
            adata.layers['X_rank'] = normalize_by_cell_area(
                adata.X,
                cell_areas=adata.obs['cell_area'].values)
            adata.layers['X_neighborhood_rank'] = normalize_by_cell_area(
                adata.X,
                cell_areas=adata.obs['neighborhood_cell_area'].values)
        else:
            if self.rank_cell_norm_method is None:
                adata.layers['X_rank'] = adata.X
                adata.layers['X_neighborhood_rank'] = adata.layers['X_neighborhood']
            else:
                raise ValueError(
                    f"Invalid 'cell_norm_method' {self.rank_cell_norm_method}.")

        if self.count_cell_norm_method == 'read_depth':
            adata.layers['X_count'] = normalize_by_read_depth(adata.X)
            adata.layers['X_neighborhood_count'] = normalize_by_read_depth(
                adata.layers['X_neighborhood'])

        elif self.count_cell_norm_method == 'gene_corrected_read_depth':
            adata.layers['X_count'] = normalize_by_gene_corrected_read_depth(
                adata.X)
            adata.layers['X_neighborhood_count'] = normalize_by_gene_corrected_read_depth(
                adata.layers['X_neighborhood']) 

        elif self.count_cell_norm_method == 'cell_area':
            adata.layers['X_count'] = normalize_by_cell_area(
                adata.X,
                cell_areas=adata.obs['cell_area'].values)
            adata.layers['X_neighborhood_count'] = normalize_by_cell_area(
                adata.X,
                cell_areas=adata.obs['neighborhood_cell_area'].values)
        else:
            if self.count_cell_norm_method is None:
                adata.layers['X_count'] = adata.X
                adata.layers['X_neighborhood_count'] = adata.layers['X_neighborhood']
            else:
                raise ValueError(
                    f"Invalid 'cell_norm_method' {self.count_cell_norm_method}.")

        # Perform normalization of counts per gene for rank and count
        # tokenization
        if self.rank_gene_norm_method == 'mean':
            if self.rank_cell_norm_method is None:
                norm_factor = 'mean'
            elif self.rank_cell_norm_method == 'read_depth':
                norm_factor = 'read_depth_mean'
            elif self.rank_cell_norm_method == 'gene_corrected_read_depth':
                norm_factor = 'gene_corrected_read_depth_mean'
            elif self.rank_cell_norm_method == 'cell_area':
                norm_factor = 'cell_area_mean'
            adata.layers['X_rank'] = normalize_by_factor(
                adata.layers['X_rank'],
                norm_factor_file_path=self.norm_factor_file_path,
                probed_genes=adata.var['ensembl_id'],
                norm_factor=norm_factor)
            adata.layers['X_neighborhood_rank'] = normalize_by_factor(
                adata.layers['X_neighborhood_rank'],
                norm_factor_file_path=self.norm_factor_file_path,
                probed_genes=adata.var['ensembl_id'],
                norm_factor=f"{norm_factor}_neighborhood")
        elif self.rank_gene_norm_method == 'nonzero_mean':
            if self.rank_cell_norm_method is None:
                norm_factor = 'nonzero_mean'
            elif self.rank_cell_norm_method == 'read_depth':
                norm_factor = 'read_depth_nonzero_mean'
            elif self.rank_cell_norm_method == 'gene_corrected_read_depth':
                norm_factor = 'gene_corrected_read_depth_nonzero_mean'
            elif self.rank_cell_norm_method == 'cell_area':
                norm_factor = 'cell_area_nonzero_mean'
            adata.layers['X_rank'] = normalize_by_factor(
                adata.layers['X_rank'],
                norm_factor_file_path=self.norm_factor_file_path,
                probed_genes=adata.var['ensembl_id'],
                norm_factor=norm_factor)
            adata.layers['X_neighborhood_rank'] = normalize_by_factor(
                adata.layers['X_neighborhood_rank'],
                norm_factor_file_path=self.norm_factor_file_path,
                probed_genes=adata.var['ensembl_id'],
                norm_factor=f"{norm_factor}_neighborhood")
        elif self.rank_gene_norm_method == 'seurat_v3':
            adata.layers['X_rank'] = normalize_by_seurat(adata.layers['X_rank'])
            adata.layers['X_neighborhood_rank'] = normalize_by_seurat(
                adata.layers['X_neighborhood_rank'])
        else:
            if self.rank_gene_norm_method is None:
                pass
            else:
                raise ValueError(
                    f"Invalid 'gene_norm_method' {self.rank_gene_norm_method}.")

        if self.count_gene_norm_method == 'mean':
            if self.count_cell_norm_method is None:
                norm_factor = 'mean'
            elif self.count_cell_norm_method == 'read_depth':
                norm_factor = 'read_depth_mean'
            elif self.count_cell_norm_method == 'gene_corrected_read_depth':
                norm_factor = 'gene_corrected_read_depth_mean'
            elif self.count_cell_norm_method == 'cell_area':
                norm_factor = 'cell_area_mean'
            adata.layers['X_count'] = normalize_by_factor(
                adata.layers['X_count'],
                norm_factor_file_path=self.norm_factor_file_path,
                probed_genes=adata.var['ensembl_id'],
                norm_factor=norm_factor)
            adata.layers['X_neighborhood_count'] = normalize_by_factor(
                adata.layers['X_neighborhood_count'],
                norm_factor_file_path=self.norm_factor_file_path,
                probed_genes=adata.var['ensembl_id'],
                norm_factor=f"{norm_factor}_neighborhood")
        elif self.count_gene_norm_method == 'nonzero_mean':
            if self.count_cell_norm_method is None:
                norm_factor = 'nonzero_mean'
            elif self.count_cell_norm_method == 'read_depth':
                norm_factor = 'read_depth_nonzero_mean'
            elif self.count_cell_norm_method == 'gene_corrected_read_depth':
                norm_factor = 'gene_corrected_read_depth_nonzero_mean'
            elif self.count_cell_norm_method == 'cell_area':
                norm_factor = 'cell_area_nonzero_mean'
            adata.layers['X_count'] = normalize_by_factor(
                adata.layers['X_count'],
                norm_factor_file_path=self.norm_factor_file_path,
                probed_genes=adata.var['ensembl_id'],
                norm_factor=norm_factor)
            adata.layers['X_neighborhood_count'] = normalize_by_factor(
                adata.layers['X_neighborhood_count'],
                norm_factor_file_path=self.norm_factor_file_path,
                probed_genes=adata.var['ensembl_id'],
                norm_factor=f"{norm_factor}_neighborhood")
        elif self.count_gene_norm_method == 'seurat_v3':
            adata.layers['X_count'] = normalize_by_seurat(
                adata.layers['X_count'])
        else:
            if self.count_gene_norm_method is None:
                pass
            else:
                raise ValueError(
                    f"Invalid 'gene_norm_method' {self.count_gene_norm_method}.")

        # Perform normalization of counts for rank and count tokenization
        if self.rank_count_norm_method == 'analytic_pearson_residuals':
            if (self.rank_cell_norm_method is not None) or (
                self.rank_gene_norm_method is not None):
                raise ValueError('Invalid combination of norm methods.')
            adata.layers['X_rank'] = normalize_by_analytic_pearson_residuals(
                adata.layers['X_rank'])
            adata.layers['X_neighborhood_rank'] = normalize_by_analytic_pearson_residuals(
                adata.layers['X_neighborhood_rank'])
        elif self.rank_count_norm_method == 'shifted_log':
            adata.layers['X_rank'] = normalize_by_shifted_log(
                adata.layers['X_rank'])
            adata.layers['X_neighborhood_rank'] = normalize_by_shifted_log(
                adata.layers['X_neighborhood_rank'])
        else:
            if self.rank_count_norm_method is None:
                pass
            else:
                raise ValueError(
                    f"Invalid 'counts_norm_method': {self.rank_count_norm_method}.")

        if self.count_count_norm_method == 'analytic_pearson_residuals':
            if (self.count_cell_norm_method is not None) or (
                self.count_gene_norm_method is not None):
                raise ValueError('Invalid combination of norm methods.')
            adata.layers['X_count'] = normalize_by_analytic_pearson_residuals(
                adata.layers['X_count'])
            adata.layers['X_neighborhood_count'] = normalize_by_analytic_pearson_residuals(
                adata.layers['X_neighborhood_count'])
        elif self.count_count_norm_method == 'shifted_log':
            adata.layers['X_count'] = normalize_by_shifted_log(
                adata.layers['X_count'])
            adata.layers['X_neighborhood_count'] = normalize_by_shifted_log(
                adata.layers['X_neighborhood_count'])
        else:
            if self.count_count_norm_method is None:
                pass
            else:
                raise ValueError(
                    f"Invalid 'counts_norm_method': {self.count_count_norm_method}.")

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
            if self.include_zero_expr_genes:
                norm_counts_cell_rank = adata[
                    i : i + self.chunk_size, coding_miRNA_idx].layers['X_rank'].toarray()
                norm_counts_neighborhood_rank = adata[
                    i : i + self.chunk_size, coding_miRNA_idx].layers[
                        'X_neighborhood_rank'].toarray()
                norm_counts_cell_count = adata[
                    i : i + self.chunk_size, coding_miRNA_idx].layers['X_count'].toarray()
                norm_counts_neighborhood_count = adata[
                    i : i + self.chunk_size, coding_miRNA_idx].layers[
                        'X_neighborhood_count'].toarray()

                # Rank gene tokens and append across chunks
                adata_dict['gene_tokens_cell'] += [
                    rank_gene_tokens(norm_counts_cell_rank[j],
                    coding_miRNA_tokens_cell)
                    for j in range(norm_counts_cell_rank.shape[0])]
                adata_dict['gene_tokens_neighborhood'] += [
                    rank_gene_tokens(norm_counts_neighborhood_rank[j],
                    coding_miRNA_tokens_neighborhood)
                    for j in range(norm_counts_neighborhood.shape[0])]

                # Rank gene expression and append across chunks
                adata_dict['gene_expr_cell'] += [
                    norm_counts_cell_count[j][np.argsort(-norm_counts_cell_rank[j])]
                    for j in range(norm_counts_cell_rank.shape[0])]
                adata_dict['gene_expr_neighborhood'] += [
                    norm_counts_neighborhood_count[j][
                        np.argsort(-norm_counts_neighborhood_rank[j])]
                    for j in range(norm_counts_neighborhood_rank.shape[0])]
            else:
                norm_counts_cell_rank = sp.csr_matrix(adata[
                    i : i + self.chunk_size, coding_miRNA_idx].layers['X_rank'])
                norm_counts_neighborhood_rank = sp.csr_matrix(adata[
                    i : i + self.chunk_size, coding_miRNA_idx].layers[
                        'X_neighborhood_rank'])
                norm_counts_cell_count = sp.csr_matrix(adata[
                    i : i + self.chunk_size, coding_miRNA_idx].layers['X_count'])
                norm_counts_neighborhood_count = sp.csr_matrix(adata[
                    i : i + self.chunk_size, coding_miRNA_idx].layers[
                        'X_neighborhood_count'])

                # Rank gene tokens and append across chunks
                adata_dict['gene_tokens_cell'] += [
                    rank_gene_tokens(norm_counts_cell_rank[j].data,
                    coding_miRNA_tokens_cell[norm_counts_cell_rank[j].indices])
                    for j in range(norm_counts_cell_rank.shape[0])]
                adata_dict['gene_tokens_neighborhood'] += [
                    rank_gene_tokens(norm_counts_neighborhood_rank[j].data,
                    coding_miRNA_tokens_neighborhood[
                        norm_counts_neighborhood_rank[j].indices])
                    for j in range(norm_counts_neighborhood_rank.shape[0])]

                # Rank gene expression and append across chunks
                adata_dict['gene_expr_cell'] += [
                    norm_counts_cell_count[j].data[np.argsort(
                        -norm_counts_cell_rank[j].data)]
                    for j in range(norm_counts_cell_rank.shape[0])]
                adata_dict['gene_expr_neighborhood'] += [
                    norm_counts_neighborhood_count[j].data[
                        np.argsort(-norm_counts_neighborhood_rank[j].data)]
                    for j in range(norm_counts_neighborhood_rank.shape[0])]

        # Add cell IDs for collecting metadata at inference time
        adata_dict['cell_id'] = adata.obs['cell_id'].values.tolist()
        
        n_cells = len(adata)

        # Add read depth and number of genes
        #adata_dict['cell_total_counts'] = [
        #    [total_counts, total_neighborhood_counts] for 
        #    total_counts, total_neighborhood_counts in zip(
        #        adata.X.sum(axis=1).A1.tolist(),
        #        adata.layers['X_neighborhood'].sum(axis=1).A1.tolist())]
        #adata_dict['cell_n_probed_genes'] = [adata.X.shape[1]] * n_cells

        # Add special tokens (positional tokens and value tokens)
        batch_id_key = f"{adata.uns['dataset_id']}_{adata.uns['batch']}"

        adata_dict['batch_token'] = [self.token_dict['spt_batch']] * n_cells
        adata_dict['gene_panel_token'] = [
            self.token_dict['spt_gene_panel']] * n_cells
        adata_dict['assay_token'] = [self.token_dict['spt_assay']] * n_cells
        adata_dict['species_token'] = [self.token_dict['spt_species']] * n_cells
        adata_dict['tissue_token'] = [self.token_dict['spt_tissue']] * n_cells

        #adata_dict['batch_value_token'] = [
        #    self.token_dict[f'spv_{batch_id_key}']] * n_cells
        #adata_dict['gene_panel_value_token'] = [self.token_dict[
        #    f'spv_gene_panel{len(adata.var_names)}']
        #    ] * n_cells
        #adata_dict['assay_value_token'] = [
        #    self.token_dict[f'spv_{adata.uns["assay"]}']] * n_cells
        #adata_dict['species_value_token'] = [
        #    self.token_dict[f'spv_{adata.uns["species"]}']] * n_cells
        #adata_dict['tissue_value_token'] = [
        #    self.token_dict[f'spv_{adata.uns["tissue"]}']] * n_cells

        # Store values with right embedding index for count tokenizer
        # Leave space for <pad>, <mask> and <cls> tokens
        spv_dict = {
            k: v for k, v in self.token_dict.items() if k.startswith('spv_')}
        spv_start_idx = min(spv_dict.values())
        spv_idx_subtract = (spv_start_idx - 2 - self.max_cls_tokens)

        adata_dict['batch_value'] = [
            self.token_dict[f'spv_{batch_id_key}'] - spv_idx_subtract] * n_cells
        adata_dict['gene_panel_value'] = [self.token_dict[
            f'spv_gene_panel{len(adata.var_names)}']
            - spv_idx_subtract] * n_cells
        adata_dict['assay_value'] = [
            self.token_dict[
                f'spv_{adata.uns["assay"]}'] - spv_idx_subtract] * n_cells
        adata_dict['species_value'] = [self.token_dict[
            f'spv_{adata.uns["species"]}'] - spv_idx_subtract] * n_cells
        adata_dict['tissue_value'] = [self.token_dict[
            f'spv_{adata.uns["tissue"]}'] - spv_idx_subtract] * n_cells

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
        gene_tokens_neighborhood, \
        n_nonzero_neighborhood_tokens = process_gene_tokens(
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
        #example['cls_tokens'] = [self.token_dict['<cls_0>'],
        #                         self.token_dict['<cls_1>']]
        example['assay_token'] = [example['assay_token']]
        example['species_token'] = [example['species_token']]
        example['tissue_token'] = [example['tissue_token']]
        example['gene_panel_token'] = [example['gene_panel_token']]
        example['batch_token'] = [example['batch_token']]

        #example['assay_value_token'] = [example['assay_value_token']]
        #example['species_value_token'] = [example['species_value_token']]
        #example['tissue_value_token'] = [example['tissue_value_token']]
        #example['gene_panel_value_token'] = [example['gene_panel_value_token']]
        #example['batch_value_token'] = [example['batch_value_token']]        

        # Retrieve special token values
        example['assay_value'] = [example['assay_value']]
        example['species_value'] = [example['species_value']]
        example['tissue_value'] = [example['tissue_value']]
        example['gene_panel_value'] = [example['gene_panel_value']]
        example['batch_value'] = [example['batch_value']]

        return example