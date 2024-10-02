from typing import List, Literal, Optional, Tuple, Union

import datasets
import numpy as np
import torch
from torch.utils.data import Dataset


class CellBaseDataset(Dataset):
    def __init__(self,
                 dataset: datasets.Dataset,
                 vocab_size: int,
                 seq_len_cell: int,
                 seq_len_neighborhood: int,
                 special_tokens: List=[
                    'cls_cell',
                    'cls_neighborhood',
                    'assay',
                    'species',
                    'tissue',
                    'gene_panel',
                    'batch'],
                 sampling_strategy: Optional[
                    Literal['norm_count_rank_sampling',
                            'norm_count_rank_sampling_rep',
                            'rand_sampling',
                            'rand_sampling_rep']]=None,
                 ):
        """
        Torch CellBaseDataset class.

        Parameters
        -----------
        dataset:
            Huggingface dataset with sequences of gene tokens and special
            tokens.
        vocab_size:
            Size of the vocabulary.
        seq_len_cell:
            Sequence length of the cell tokens.
        seq_len_neighborhood:
            Sequence length of the neighborhood tokens.
        special_tokens:
            Special tokens to be included in the sequence processed by the
            model.
        sampling_strategy:
            Token sampling strategy.
        """
        self.dataset = dataset
        self.len = len(self.dataset)
        self.n_nonzero_tokens = self.dataset["n_nonzero_tokens"]
        self.vocab_size = vocab_size
        self.seq_len_cell = seq_len_cell
        self.seq_len_neighborhood = seq_len_neighborhood
        self.n_special_tokens = len(special_tokens)
        self.seq_len = (seq_len_cell +
                        seq_len_neighborhood +
                        self.n_special_tokens)
        self.special_tokens = special_tokens
        self.sampling_strategy = sampling_strategy

    def __len__(self):
        return self.len

    def _add_special_tokens_to_seq(self,
                                   seq_tokens: List,
                                   seg_tokens: List,
                                   item: int,
                                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add special tokens to sequence and update segment tokens.

        Parameters
        -----------
        seq_tokens:
            Token sequence including all segments.
        seg_tokens:
            Segment tokens including all segments.
        item:
            Index of the cell in the huggingface dataset.

        Returns
        -----------
        seq_tokens:
            Sequence tokens with special tokens included at sequence start.
        seg_tokens:
            Segment tokens with 0s for special tokens at sequence start.
        """
        if 'batch' in self.special_tokens:
            seq_tokens = self.dataset[item]["batch_token"] + seq_tokens
        if 'gene_panel' in self.special_tokens:
            seq_tokens = self.dataset[item]["gene_panel_token"] + seq_tokens
        if 'tissue' in self.special_tokens:
            seq_tokens = self.dataset[item]["tissue_token"] + seq_tokens
        if 'species' in self.special_tokens:
            seq_tokens = self.dataset[item]["species_token"] + seq_tokens
        if 'assay' in self.special_tokens:
            seq_tokens = self.dataset[item]["assay_token"] + seq_tokens
        if 'cls_neighborhood' in self.special_tokens:
            seq_tokens = self.dataset[item][
                "cls_neighborhood_token"] + seq_tokens
        if 'cls_cell' in self.special_tokens:
            seq_tokens = self.dataset[item][
                "cls_cell_token"] + seq_tokens
        seg_tokens = [0] * self.n_special_tokens + seg_tokens

        return seq_tokens, seg_tokens

    def _create_sampled_token_seq(self,
                                  tokens: List,
                                  n_nonzero_tokens: int,
                                  size: int,
                                  ) -> List[int]:
            """
            Sample a subset of tokens based on a sampling strategy.

            Parameters
            -----------
            tokens:
                List of tokens.
            n_nonzero_tokens:
                Number of nonzero tokens in `tokens`.
            size:
                Size of the sampled subset.
                
            Returns
            --------
            sampled_tokens:
                List of sampled tokens.
            """
            if 'norm_count_rank_sampling' in self.sampling_strategy:
                # Calculate weights based on rank and number of nonzero tokens: the
                # higher the rank, the higher the weight
                # seq = [4, 1, 3, 2, 5, 0, 0, 0]
                # n_nonzero_tokens = 5  
                # sum_rank = 5 * (5 + 1) / 2.0 = 15.0
                # weights = [(n_nonzero_tokens - i)/sum_rank for i in range(
                #     n_nonzero_tokens)] 
                # = [0.333, 0.266, 0.2, 0.133, 0.066]
                # np.sum(weights) = 1.0
                sum_rank = (n_nonzero_tokens * (n_nonzero_tokens + 1) / 2.0) + 1e-9
                weights = [(n_nonzero_tokens - i)/sum_rank for i in range(
                    n_nonzero_tokens)]
                assert np.isclose(np.sum(weights), 1.0)
            elif 'rand_sampling' in self.sampling_strategy:
                weights = np.ones(n_nonzero_tokens) / n_nonzero_tokens
            else:
                raise ValueError(
                    f"'{self.sampling_strategy}' is an invalid sampling strategy.")
                
            # Sample token indices based on weights
            sampled_indices = np.random.choice(
                np.arange(n_nonzero_tokens),
                size=min(size, n_nonzero_tokens),
                p=weights,
                replace=(True if self.sampling_strategy is not None and 'rep'
                     in self.sampling_strategy else False))
                
            # Sort sampled indices to preserve rank order
            sampled_indices = np.sort(sampled_indices)
            sampled_tokens = [tokens[i] for i in sampled_indices]

            if size > n_nonzero_tokens:
                sampled_tokens.extend([0] * (size - len(sampled_tokens)))
            
            return sampled_tokens
         
    def _get_gene_tokens_for_segment(self, 
                                     item: int,
                                     segment_idx: int,
                                     segment_seq_len: int,
                                     ) -> List[int]:
        """
        Get gene tokens for a given segment based on sampling strategy.

        Parameters
        -----------
        item:
            Index of the cell in the huggingface dataset.
        segment_idx:
            Index of the segment for which tokens are retrieved.
        segment_seq_len:
            Desired length of the segment token sequence.

        Returns
        --------
        segment_gene_tokens:
            List of tokens for a given segment.
        """
        # Only keep gene tokens in specified segment
        segment_gene_tokens = [gene_token for gene_token, seg_token in zip(
            self.dataset[item]["gene_tokens"], self.dataset[item]["seg_tokens"])
            if seg_token == segment_idx]

        # Validate that segment sequence length is specified correctly
        if (self.sampling_strategy is not None and 'rep' in
        self.sampling_strategy):
            pass
        else:
            if segment_seq_len > len(segment_gene_tokens):
                raise ValueError(
                    'Sequence length for a given segment cannot be larger than'
                    'segment size when not sampling with replacement.')

        # If no sampling strategy is specified, use all tokens up to specified
        # length
        if self.sampling_strategy is None:
            segment_gene_tokens = segment_gene_tokens[:segment_seq_len]
        # Otherwise, sample a subset of tokens based on the sampling strategy
        else:
            segment_n_nonzero_tokens = sum(
                1 for token in segment_gene_tokens if token != 0)

            segment_gene_tokens = self._create_sampled_token_seq(
                tokens=segment_gene_tokens,
                n_nonzero_tokens=segment_n_nonzero_tokens,
                size=segment_seq_len)
                
        return segment_gene_tokens


class CellGraphDataset(CellBaseDataset):
    def __init__(self,
                 **base_dataset_kwargs
                 ):
        """
        Torch CellGraphDataset class.

        Parameters
        -----------
        **base_dataset_kwargs:
            Keyword arguments for the initialization of CellBaseDataset.
        """
        super().__init__(**base_dataset_kwargs)
         
    def __getitem__(self,
                    item: int
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get (sampled) gene tokens
        gene_tokens_index_cell = self._get_gene_tokens_for_segment(
            item=item,
            segment_idx=1, # index cell seg
            segment_seq_len=self.seq_len_index_cell)
        seg_tokens = [1] * len(gene_tokens_index_cell)
        gene_tokens_neighbor_cells = []
        for segment_idx in np.unique(self.dataset[item]["seg_tokens"]):
            if segment_idx != 1:
                segment_gene_tokens = self._get_gene_tokens_for_segment(
                    item=item,
                    segment_idx=segment_idx, # neighbor cell segs
                    segment_seq_len=self.seq_len_neighbor_cells)
                gene_tokens_neighbor_cells.extend(segment_gene_tokens)
                seg_tokens.extend([segment_idx] * len(segment_gene_tokens))
        seq_tokens = gene_tokens_index_cell + gene_tokens_neighbor_cells

        # Add special tokens
        seq_tokens, seg_tokens = self._add_special_tokens_to_seq(
            seq_tokens=seq_tokens,
            seg_tokens=seg_tokens,
            item=item)

        seq_tokens = torch.tensor(seq_tokens)
        seg_tokens = torch.tensor(seg_tokens).int()

        return seq_tokens, seg_tokens, self.dataset[item]["cell_id"]


class CellNeighborhoodDataset(CellBaseDataset):
    def __init__(self,
                 **base_dataset_kwargs
                 ):
        """
        Torch CellNeighborhoodDataset class.

        Parameters
        -----------
        **base_dataset_kwargs:
            Keyword arguments for the initialization of CellBaseDataset.
        """
        super().__init__(**base_dataset_kwargs)

    def __getitem__(self,
                    item: int
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get (sampled) gene tokens
        gene_tokens_cell = self._get_gene_tokens_for_segment(
            item=item,
            segment_idx=1, # cell seg
            segment_seq_len=self.seq_len_cell)
        gene_tokens_neighborhood = self._get_gene_tokens_for_segment(
            item=item,
            segment_idx=2, # neighborhood seg
            segment_seq_len=self.seq_len_neighborhood)
        seq_tokens = gene_tokens_cell + gene_tokens_neighborhood
        seg_tokens = [1] * len(gene_tokens_cell) + [2] * len(
            gene_tokens_neighborhood)

        # Add special tokens
        seq_tokens, seg_tokens = self._add_special_tokens_to_seq(
            seq_tokens=seq_tokens,
            seg_tokens=seg_tokens,
            item=item)

        seq_tokens = torch.tensor(seq_tokens)
        seg_tokens = torch.tensor(seg_tokens).int()

        return seq_tokens, seg_tokens, self.dataset[item]["cell_id"]


def make_cell_dataset(tokenizer_type: Literal['cell_graph',
                                              'cell_neighborhood'],
                      **cell_dataset_kwargs
                      ) -> Union[CellGraphDataset, CellNeighborhoodDataset]:
    """
    Based on tokenizer type, return CellGraphDataset or CellNeighborhoodDataset.
    """
    if tokenizer_type == 'cell_graph':
        cell_dataset = CellGraphDataset(**cell_dataset_kwargs)
    elif tokenizer_type  == 'cell_neighborhood':
        cell_dataset = CellNeighborhoodDataset(**cell_dataset_kwargs)

    return cell_dataset
