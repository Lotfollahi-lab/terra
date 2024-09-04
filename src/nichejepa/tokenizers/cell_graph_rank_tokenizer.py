"""
Cell Graph Rank Tokenizer.

Adapted from Theodoris, C. V. et al. Transfer learning enables predictions in network biology. Nature 618, 616–624
(2023); https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/tokenizer.py (12.04.2024).

Input Data
----------
Required format:
    Raw counts spatial transcriptomics (ST) data with all genes (no feature selection) as '.h5ad' (anndata) files.
    Spatial coordinates are stored in adata.obsm["spatial"].
Required gene attributes:
    Ensembl ID for each gene ('ensembl_id').
Optional cell attributes:
    Binary indicator of whether cell should be used for tokenization based on user-defined filtering criteria
    ('filter_pass').
    Any other cell metadata can be passed on to the tokenized dataset as a custom attribute dictionary.

Usage
----------
.. code-block :: python
    >>> from nichejepa import CellGraphRankTokenizer
    >>> tk = CellGraphRankTokenizer(custom_attr_name_dict={"cell_type": "cell_types"}, nproc=4)
    >>> tk.tokenize_data("input_directory", "output_directory", "output_file_prefix")

Description
----------
Input data is a directory with '.h5ad' files containing raw counts from ST data, including all genes detected without
feature selection. The input file type is specified by the argument 'file_format' in the tokenize_data function. Genes
should be labeled with Ensembl IDs (adata.var['ensembl_id']), which provide a unique identifer for conversion to tokens.
Gene names can be converted to Ensembl IDs via the helper function nichejepa.utils.genes.get_ensembl_ids() or via the
pyensembl Python package. No cell metadata is required, but custom cell attributes may be passed onto the tokenized
dataset by providing a dictionary of custom attributes, which is formatted as
{original_attr_name: desired_dataset_attr_name}. For example, if the original '.h5ad' file has cell attributes in
adata.obs["cell_type"] and one would like to retain these attributes as labels in the tokenized dataset with the new
names "cell_types", the following custom attribute dictionary should be provided: {"cell_type": "cell_types"}.
Additionally, if the original '.h5ad' file contains a cell attribute called adata.obs["filter_pass"], this will be used
as a binary indicator of whether to include these cells in the tokenization. All cells with "1" in this attribute will
be tokenized, whereas the others will be excluded. One may use this column to indicate QC filtering or other criteria
for selection for inclusion in the final tokenized dataset. If one's data is in other formats besides '.h5ad', one can
use the relevant tools (such as Anndata tools) to convert the file to '.h5ad' format prior to initializing the
CellGraphRankTokenizer.
"""

from __future__ import annotations

import logging
import pickle
import warnings
import concurrent
from pathlib import Path
from typing import Literal, Optional, Tuple

import anndata as ad
import numpy as np
import scipy.sparse as sp
import squidpy as sq
from datasets import Dataset

from ..normalizers.normalize_by_analytic_pearson_residuals import normalize_by_analytic_pearson_residuals
from ..normalizers.normalize_by_cell_area import normalize_by_cell_area
from ..normalizers.normalize_by_mean import normalize_by_mean
from ..normalizers.normalize_by_nonzero_mean import normalize_by_nonzero_mean
from ..normalizers.normalize_by_read_depth import normalize_by_read_depth
from ..normalizers.normalize_by_seurat import normalize_by_seurat
from ..normalizers.normalize_by_shifted_log_mean import normalize_by_shifted_log_mean
from ..normalizers.normalize_by_shifted_log import normalize_by_shifted_log
from ..preprocessors.filter_poor_quality_cells import filter_poor_quality_cells
from .tokenize import process_gene_tokens, rank_gene_tokens


warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*") # noqa

logger = logging.getLogger(__name__)

GENE_MEANS_FILE = Path(__file__).parent.parent.parent.parent / "cell_gene_means_dictionary.pkl"
GENE_NZMEANS_FILE = Path(__file__).parent.parent.parent.parent / "cell_gene_nzmeans_dictionary.pkl"
GENE_LOGMEANS_FILE = Path(__file__).parent.parent.parent.parent / "cell_gene_logmeans_dictionary.pkl"
TOKEN_DICTIONARY_FILE = Path(__file__).parent.parent.parent.parent / "token_dictionary.pkl"


class CellGraphRankTokenizer:
    def __init__(self,
                 custom_attr_name_dict: Optional[dict] = None,
                 nproc: int = 1,
                 processing_mode: Optional[Literal["sequential", "parallel"]] = "sequential",                 
                 chunk_size: int = 512,
                 model_input_size: int = 2048,
                 tokens_per_cell: int = 64,
                 norm_method: Literal["analytic_pearson_residuals",
                                      "mean",
                                      "nzmedian",
                                      "seurat_v3",
                                      "shifted_log_mean"
                                      "shifted_log"]="seurat_v3",
                 norm_factor: Optional[Literal["read_depth", "cell_area"]]=None,
                 gene_means_file: Path | str = GENE_MEANS_FILE,
                 gene_nzmeans_file: Path | str = GENE_NZMEANS_FILE,
                 gene_logmeans_file: Path | str = GENE_LOGMEANS_FILE,
                 token_dictionary_file: Path | str = TOKEN_DICTIONARY_FILE,
                 special_tokens: Optional[list[str]] = None, # ["<cls>"],
                 special_tokens_idx: Optional[list[int]] = None #[0]
                ):
        """
        Initialize spatial transcriptomics rank tokenizer.

        Parameters
        ----------
        custom_attr_name_dict:
            Dictionary of custom attributes to be added to the Hugging Face dataset. Keys are the names of the
            attributes in the '.h5ad' files. Values are the names of the attributes in the Hugging Face dataset.
        nproc
            Number of processes to use for dataset mapping.
        processing_mode:
            Processing mode for tokenizing '.h5ad' files. Can be 'sequential' or 'parallel'.            
        chunk_size:
            Chunk size for adata tokenizer.
        model_input_size:
            Max input size of the model to truncate input to.
        norm_method:
            Normalization method used for count normalization before ranking.
        norm_factor:
            Normalization factor for cellular normalization to adjust for cell size differences.
            Is not used if 'norm_method' is 'analytic_pearson_residuals'.
        gene_means_file:
            Path to pickle file containing dictionary of mean gene expression of cells across STcorpus (for each gene).
            Only relevant if 'norm_method' in ['mean'].
        gene_nzmeans_file:
            Path to pickle file containing dictionary of non-zero mean gene expression of cells across STcorpus (for
            each gene).
            Only relevant if 'norm_method' in ['nzmean'].
        gene_logmeans_file:
            Path to pickle file containing dictionary of log mean gene expression of cells across STcorpus (for each
            gene).
            Only relevant if 'norm_method' in ['shifted_logmean'].
        token_dictionary_file:
            Path to pickle file containing token dictionary (gene tokens are Ensembl IDs).
        special_tokens:
            List with special tokens inserted into the gene token vector containing cell gene tokens.
        special_tokens_idx:
            Index where special tokens are to be inserted into the cell gene token vector.
        """

        self.custom_attr_name_dict = custom_attr_name_dict
        self.nproc = nproc
        self.processing_mode = processing_mode
        self.chunk_size = chunk_size
        self.model_input_size = model_input_size
        self.tokens_per_cell = tokens_per_cell
        self.norm_method = norm_method
        self.norm_factor = norm_factor
        self.gene_means_file = gene_means_file
        self.gene_nzmeans_file = gene_nzmeans_file
        self.gene_logmeans_file = gene_logmeans_file
        self.special_tokens = special_tokens
        self.special_tokens_idx = special_tokens_idx

        # Load token dictionary
        with open(token_dictionary_file, "rb") as f:
            self.token_dict = pickle.load(f)

        # Get vocabulary and gene Ensembl IDs (protein-coding and miRNA genes)
        self.vocab = list(self.token_dict.keys())
        self.coding_miRNA_ids = [key.split("_")[0] for key in list(self.vocab) if "ENS" in key]
        self.coding_miRNA_dict = dict(zip(self.coding_miRNA_ids, [True] * len(self.vocab)))

    def tokenize_data(self,
                      input_directory: Path | str,
                      output_directory: Path | str,
                      output_file_prefix: str,
                      file_format: Literal["h5ad"] = "h5ad",
                      use_generator: bool = False):
        """
        Tokenize files in 'input_directory' and save as tokenized '.dataset' file in 'output_directory'.

        Parameters
        ----------
        input_directory:
            Path to directory containing '.h5ad' (anndata) files.
        output_directory:
            Path to directory where tokenized data will be saved as '.dataset' file.
        output_file_prefix:
            Prefix for output file.
        file_format:
            Format of input files. Can be '.h5ad'.
        use_generator:
            If 'True', use generator for tokenization, else dict.
        """

        gene_tokens, cell_pos_tokens, gene_pos_tokens, cell_metadata = self.tokenize_files(
            Path(input_directory), file_format
            )

        tokenized_dataset = self.create_dataset(gene_tokens,
                                                cell_pos_tokens,
                                                gene_pos_tokens,
                                                cell_metadata,
                                                use_generator=use_generator)

        output_path = str((Path(output_directory) / output_file_prefix).with_suffix(".dataset"))
        tokenized_dataset.save_to_disk(output_path)

    def tokenize_files(self,
                       data_directory: Path | str,
                       file_format: Literal["h5ad"] = "h5ad") -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Tokenize multiple files.

        Parameters
        ----------
        data_directory:
            Path to the directory containing the files to be tokenized.
        file_format:
            Format of the files to be tokenized.

        Returns
        ----------
        gene_tokens:
            Cell-wise vector of ranked gene tokens.
        cell_pos_tokens:
            Cell-wise vector of positional tokens for cells.
        gene_pos_tokens:
            Cell-wise vector of positional tokens for genes.
        cell_metadata:
            Dictionary of cell metadata where keys are metadata columns and values are lists of cell-wise values.
        """

        gene_tokens = []
        cell_pos_tokens = []
        gene_pos_tokens = []
        if self.custom_attr_name_dict is not None:
            cell_attr = [attr_key for attr_key in self.custom_attr_name_dict.keys()]
            cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.values()}

        file_found = 0

        tokenize_file_fn = self.tokenize_adata

        if self.processing_mode == "sequential":
        # Loop through data directory to tokenize '.h5ad' files sequentially
            print("Tokenizing files sequentially...")
            for file_path in data_directory.glob(f"*.{file_format}"):
                file_found = 1
                print(f"Tokenizing '{file_path}'...")
                file_gene_tokens, file_cell_pos_tokens, file_gene_pos_tokens, file_cell_metadata = tokenize_file_fn(file_path)
                gene_tokens += file_gene_tokens
                cell_pos_tokens += file_cell_pos_tokens
                gene_pos_tokens += file_gene_pos_tokens
                if self.custom_attr_name_dict is not None:
                    for k in cell_attr:
                        cell_metadata[self.custom_attr_name_dict[k]] += file_cell_metadata[k]
                else:
                    cell_metadata = None
        elif self.processing_mode == "parallel":
            print("Tokenizing files in parallel...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.nproc) as executor:
                futures = []
                for file_path in data_directory.glob(f"*.{file_format}"):
                    file_found = 1
                    print(f"Tokenizing '{file_path}'...")
                    future = executor.submit(tokenize_file_fn, file_path)
                    futures.append(future)
                for future in concurrent.futures.as_completed(futures):
                    file_gene_tokens, file_cell_pos_tokens, file_gene_pos_tokens, file_cell_metadata = future.result()
                    gene_tokens += file_gene_tokens
                    cell_pos_tokens += file_cell_pos_tokens
                    gene_pos_tokens += file_gene_pos_tokens
                    if self.custom_attr_name_dict is not None:
                        for k in cell_attr:
                            cell_metadata[self.custom_attr_name_dict[k]] += file_cell_metadata[k]
                    else:
                        cell_metadata = None

        if file_found == 0:
            logger.error(f"No '.{file_format}' files found in directory '{data_directory}'.")
            raise

        return gene_tokens, cell_pos_tokens, gene_pos_tokens, cell_metadata

    def tokenize_adata(self,
                       adata_file_path: Path | str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Tokenize cells from an '.h5ad' (anndata) file.

        Parameters
        ----------
        adata_file_path:
            Path to anndata file containing cells to be tokenized.

        Returns
        ----------
        gene_tokens:
            Cell-wise vector of ranked cell gene tokens.
        cell_pos_tokens:
            Cell-wise vector of positional tokens for cells.
        gene_pos_tokens:
            Cell-wise vector of positional tokens for genes.
        cell_metadata:
            Dictionary of cell metadata where keys are metadata columns and values are lists of cell-wise values.
        """

        adata = ad.read_h5ad(adata_file_path)

        print("Filtering cells.")
        # Filter to remove poor quality cells
        adata = filter_poor_quality_cells(adata)

        print("Computing spatial neighborhood graph.")
        # Compute spatial neighborhood graph based on Visium spot diameter of 55 microns
        sq.gr.spatial_neighbors(adata,
                                coord_type="generic",
                                spatial_key="spatial",
                                radius=27.5)

        print("Normalizing gene expression counts.")
        # Normalize counts before gene ranking
        if self.norm_method == "analytic_pearson_residuals":
            adata.X = normalize_by_analytic_pearson_residuals(adata.X)
        elif self.norm_factor == "read_depth":
            adata.X = normalize_by_read_depth(adata.X)
        elif self.norm_factor == "cell_area":
            adata.X = normalize_by_cell_area(adata.X,
                                             cell_areas=adata.obs["cell_area"].values)

        if self.norm_method == "seurat_v3":
            adata.X = normalize_by_seurat(adata.X)
        elif self.norm_method == "mean":
            adata.X = normalize_by_mean(adata.X,
                                        gene_means_file=self.gene_means_file,
                                        probed_genes=adata.var["ensembl_id"])
        elif self.norm_method == "nzmean":
            adata.X = normalize_by_nonzero_mean(adata.X,
                                                  gene_nzmeans_file=self.gene_nzmeans_file,
                                                  probed_genes=adata.var["ensembl_id"])
        elif self.norm_method == "shifted_logmean":
            adata.X = normalize_by_shifted_log_mean(adata.X,
                                                    gene_logmeans_file=self.gene_logmeans_file,
                                                    probed_genes=adata.var["ensembl_id"])
        elif self.norm_method == "shifted_log":
            adata.X = normalize_by_shifted_log(adata.X)

        # Initialize cell metadata
        if self.custom_attr_name_dict is not None:
            cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.keys()}

        # Retrieve gene tokens for genes contained in dataset and vocab, i.e. protein-coding and miRNA genes
        coding_miRNA_idx = np.where(
            [self.coding_miRNA_dict.get(gene_id, False) for gene_id in adata.var["ensembl_id"]]
            )[0]
        coding_miRNA_ids = adata.var["ensembl_id"][coding_miRNA_idx]
        coding_miRNA_tokens = np.array([self.token_dict[gene_id] for gene_id in coding_miRNA_ids])

        gene_tokens = []
        cell_pos_tokens = []
        gene_pos_tokens = []

        print("Retrieving tokens for index cell and adding cell metadata.")
        # Divide cells into chunks and loop through chunks
        for i in range(0, len(adata), self.chunk_size):
            norm_counts = sp.csr_matrix(adata[i : i + self.chunk_size, coding_miRNA_idx].X)

            # Rank gene tokens of index cell and append across chunks
            gene_tokens_chunk = [rank_gene_tokens(norm_counts[j].data,
                                                  coding_miRNA_tokens[norm_counts[j].indices],
                                                  self.tokens_per_cell) for j in range(norm_counts.shape[0])]
            gene_tokens += gene_tokens_chunk
            # Add positional tokens for each gene token
            cell_pos_tokens += [[0] * len(gene_tokens_chunk[j]) for j in range(len(gene_tokens_chunk))]
            gene_pos_tokens += [[k for k in range(len(gene_tokens_chunk[j]))] for j in range(len(gene_tokens_chunk))]

            # Add values to cell metadata
            if self.custom_attr_name_dict is not None:
                for k in cell_metadata.keys():
                    cell_metadata[k] += adata[i : i + self.chunk_size].obs[k].tolist()
            else:
                cell_metadata = None

        print("Retrieving tokens for neighborhood cells.")
        gene_tokens_copy = gene_tokens.copy()
        # Loop through all cells to add neighbor cell gene tokens based on position of cell compared to index cell.
        # Gene tokens of cells that are closer to the index cell will be added first.
        for i in range(0, len(adata)):
            # Get sorted indices of neighbor cells based on distance to index cell
            row_start = adata.obsp["spatial_distances"].indptr[i]
            row_end = adata.obsp["spatial_distances"].indptr[i+1]
            row_data = adata.obsp["spatial_distances"].data[row_start:row_end]
            sorted_indices = np.argsort(row_data)
            # Loop through distance-sorted neighbor cells and add gene, cell pos and gene pos tokens
            for j, k in enumerate(adata.obsp["spatial_connectivities"][i].nonzero()[1][sorted_indices]):
               gene_tokens[i] = np.hstack((gene_tokens[i], gene_tokens_copy[k]))
               cell_pos_tokens[i] = np.hstack((cell_pos_tokens[i],
                                               [j+1] * len(gene_tokens_copy[k])))
               gene_pos_tokens[i] = np.hstack((gene_pos_tokens[i],
                                               [l for l in range(len(gene_tokens_copy[k]))]))

        return gene_tokens, cell_pos_tokens, gene_pos_tokens, cell_metadata

    def create_dataset(self,
                       gene_tokens: np.array,
                       cell_pos_tokens: np.array,
                       gene_pos_tokens: np.array,
                       cell_metadata: dict,
                       use_generator: bool = False,
                       add_pos_tokens: bool = True,
                       keep_original_gene_tokens: bool = False) -> Dataset:
        """
        Create a Hugging Face dataset based on tokenized cells.

        Parameters
        ----------
        gene_tokens:
            Cell-wise vector of ranked gene tokens.
        cell_pos_tokens:
            Cell-wise vector of positional tokens for cells.
        gene_pos_tokens:
            Cell-wise vector of positional tokens for genes.
        cell_metadata:
            Dictionary of cell metadata where keys are metadata columns and values are lists of cell-wise values.
        use_generator:
            If 'True', use generator for tokenization, else dict.
        add_pos_tokens:
            If 'True', add positional cell and gene tokens.
        keep_original_gene_tokens:
            If 'True', keep original gene tokens in Hugging Face dataset (before padding/truncation and addition of
            special tokens).

        Returns
        ----------
        dataset:
            Hugging Face dataset containing the tokenized cells.
        """

        print("Creating Hugging Face dataset...")
        # Create dict for Hugging Face dataset creation
        dataset_dict = {"gene_tokens": gene_tokens}
        if self.custom_attr_name_dict is not None:
            dataset_dict.update(cell_metadata)

        if add_pos_tokens:
            dataset_dict["cell_pos_tokens"] = cell_pos_tokens
            dataset_dict["gene_pos_tokens"] = gene_pos_tokens

        # Create Hugging Face dataset
        if use_generator:
            def dict_generator():
                for i in range(len(gene_tokens)):
                    yield {k: dataset_dict[k][i] for k in dataset_dict.keys()}

            dataset = Dataset.from_generator(dict_generator, num_proc=self.nproc)
        else:
            dataset = Dataset.from_dict(dataset_dict)

        def format_gene_tokens(example):
            if keep_original_gene_tokens:
                # Store original gene tokens in separate features
                example["gene_tokens_original"] = example["gene_tokens"]
                example["gene_tokens_original_length"] = len(example["gene_tokens"])

            example["input_ids"] = process_gene_tokens(example["gene_tokens"],
                                                       self.model_input_size,
                                                       self.token_dict,
                                                       self.special_tokens,
                                                       self.special_tokens_idx)

            return example

        formatted_dataset = dataset.map(
            format_gene_tokens, num_proc=self.nproc)

        return formatted_dataset
