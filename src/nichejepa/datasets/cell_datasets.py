from typing import Literal, Optional, Tuple, Union

import datasets
import numpy as np
import torch
from torch.utils.data import Dataset


class CellBaseDataset(Dataset):
    def __init__(self,
                 gt_type: Literal['rank', 'counts'],
                 dataset: datasets.Dataset,
                 vocab_size: int,
                 seq_len_cell: int,
                 seq_len_neighborhood: int,
                 max_special_tokens: int,
                 special_tokens: list=[
                    'species',
                    'tissue',
                    'assay',
                    'gene_panel',
                    'batch'],
                 sampling_strategy: Optional[
                    Literal['norm_value_rank_sampling',
                            'norm_value_rank_sampling_rep',
                            'rand_sampling',
                            'rand_sampling_rep']]=None,
                 ):
        """
        Torch CellBaseDataset class.

        Parameters
        -----------
        gt_type:
            Gene transformer type.
        dataset:
            Hugging Face dataset with tokenized data.
        vocab_size:
            Size of the token vocabulary.
        seq_len_cell:
            Sequence length of the (index) cell tokens.
        seq_len_neighborhood:
            Sequence length of the neighborhood tokens.
        max_special_tokens:
            Maximum number of special tokens (if all special tokens are
            included; used to determine the first cell segment).
        special_tokens:
            Special tokens to be included in the token sequences.
        sampling_strategy:
            Token sampling strategy.
        """
        if gt_type not in ['rank', 'counts']:
            raise ValueError(f'Invalid "gt_type": {gt_type}.')
        
        self.gt_type = gt_type
        self.dataset = dataset
        self.len = len(self.dataset)
        self.vocab_size = vocab_size
        self.seq_len_cell = seq_len_cell
        self.seq_len_neighborhood = seq_len_neighborhood
        self.max_special_tokens = max_special_tokens
        self.special_tokens = special_tokens
        self.n_special_tokens = len(special_tokens)
        self.seq_len = (seq_len_cell +
                        seq_len_neighborhood +
                        self.n_special_tokens)
        self.sampling_strategy = sampling_strategy

        self.n_nz_tokens = self.dataset['n_nonzero_tokens']

    def __len__(self) -> int:
        return self.len

    def _add_special_seq(self,
                         item: int,
                         tokens: list[int],
                         segments: list[int],
                         positions: Optional[list[int]]=None,
                         values: Optional[list[float]]=None,
                         ) -> Union[Tuple[list[int],
                                          list[int],
                                          list[int]],
                                    Tuple[list[int],
                                          list[int],
                                          list[float]]]:
        """
        Add special tokens to sequence and update positions/values and segments.

        Parameters
        -----------
        item:
            Index of the cell in the Hugging Face dataset.
        tokens:
            Token sequence including all segments.
        segments:
            Segment labels including all segments.
        positions:
            Positions including all segments.
        values:
            Values including all segments.

        Returns
        -----------
        tokens:
            Sequence of tokens with special tokens included at sequence start.
        segments:
            Segment labels with extra segments for special tokens at sequence
            start.
        positions:
            Positions with extra positions for special tokens at sequence start.
        values:
            Values with special values corresponding to special tokens.
        """
        if 'batch' in self.special_tokens:
            if self.gt_type == 'rank':
                tokens = item["batch_value_token"] + tokens
            elif self.gt_type == 'counts':
                tokens = item["batch_token"] + tokens
                values = item["batch_value"] + values
        if 'gene_panel' in self.special_tokens:
            if self.gt_type == 'rank':
                tokens = item["gene_panel_value_token"] + tokens
            elif self.gt_type == 'counts':
                tokens = item["gene_panel_token"] + tokens
                values = item["gene_panel_value"] + values
        if 'tissue' in self.special_tokens:
            if self.gt_type == 'rank':
                tokens = item["tissue_value_token"] + tokens
            elif self.gt_type == 'counts':
                tokens = item["tissue_token"] + tokens
                values = item["tissue_value"] + values
        if 'species' in self.special_tokens:
            if self.gt_type == 'rank':
                tokens = item["species_value_token"] + tokens
            elif self.gt_type == 'counts':
                tokens = item["species_token"] + tokens
                values = item["species_value"] + values
        if 'assay' in self.special_tokens:
            if self.gt_type == 'rank':
                tokens = item["assay_value_token"] + tokens
            elif self.gt_type == 'counts':
                tokens = item["assay_token"] + tokens
                values = item["assay_value"] + values
            
        if any('cls' in token for token in self.special_tokens):
            n_cls_tokens = sum('cls' in token for token in self.special_tokens)
            cls_tokens = item["cls_tokens"][:n_cls_tokens]
            n_nz_cls_tokens = sum(1 for token in cls_tokens if token != 0)
            n_zero_cls_tokens = n_cls_tokens - n_nz_cls_tokens
            tokens = cls_tokens + tokens
            
            # Add <cls> and special token segments
            segments = list(range(1, 1 + n_nz_cls_tokens)) \
                + [0] * n_zero_cls_tokens \
                + list(range(1 + n_nz_cls_tokens, 1 + n_nz_cls_tokens + (
                    self.n_special_tokens - n_cls_tokens))) \
                + segments
            if self.gt_type == 'counts':
                # Add <cls> values
                values = list(range(2, 2 + n_nz_cls_tokens)) \
                    + [0] * n_zero_cls_tokens \
                    + values
            elif self.gt_type == 'rank':
                # Add <cls> and special token positions
                positions = list(range(1, 1 + n_nz_cls_tokens)) \
                    + [0] * n_zero_cls_tokens \
                    + list(range(1 + n_nz_cls_tokens, 1 + n_nz_cls_tokens + (
                        self.n_special_tokens - n_cls_tokens))) \
                    + positions

        else:
            # Add special token segments
            segments = list(range(1, 1 + self.n_special_tokens)) + segments
            if self.gt_type == 'rank':
                # Add special token positions
                positions = list(
                    range(1, 1 + self.n_special_tokens)) + positions

        if self.gt_type == 'rank':
            return tokens, segments, positions
        elif self.gt_type == 'counts':
            return tokens, segments, values

    def _sample_seq(self,
                    tokens: list[int],
                    values: Optional[list],
                    n_nz_tokens: int,
                    size: int,
                    ) -> Tuple[list[int], list[int]]:
        """
        Sample a subset of gene tokens and corresponding values based on a
        sampling strategy.

        Parameters
        -----------
        tokens:
            List of tokens.
        values:
            List of values.
        n_nz_tokens:
            Number of nonzero tokens in `tokens`.
        size:
            Size of the sampled subset.
            
        Returns
        --------
        sampled_tokens:
            List of sampled tokens.
        sampled_values:
            List of (corresponding) sampled values.
        """
        if 'norm_value_rank_sampling' in self.sampling_strategy:
            # Calculate weights based on rank and number of nonzero tokens:
            # the higher the rank, the higher the weight
            # seq = [4, 1, 3, 2, 5, 0, 0, 0]
            # n_nz_tokens = 5  
            # sum_rank = 5 * (5 + 1) / 2.0 = 15.0
            # weights = [(n_nz_tokens - i)/sum_rank for i in range(
            #     n_nz_tokens)] 
            # = [0.333, 0.266, 0.2, 0.133, 0.066]
            # np.sum(weights) = 1.0
            sum_rank = (n_nz_tokens * (n_nz_tokens + 1) / 2.0) + 1e-9
            weights = [(n_nz_tokens - i)/sum_rank for i in range(n_nz_tokens)]
            assert np.isclose(np.sum(weights), 1.0)
        elif 'rand_sampling' in self.sampling_strategy:
            weights = np.ones(n_nz_tokens) / n_nz_tokens
        else:
            raise ValueError(f"'{self.sampling_strategy}' is invalid.")
            
        # Sample token indices based on weights
        sampled_indices = np.random.choice(
            np.arange(n_nz_tokens),
            size=min(size, n_nz_tokens),
            p=weights,
            replace=(True if 'rep' in self.sampling_strategy else False))
            
        # Sort sampled indices to preserve rank order
        sampled_indices = np.sort(sampled_indices)
        sampled_tokens = [tokens[i] for i in sampled_indices]
        if self.gt_type == 'counts':
            sampled_values = [values[i] for i in sampled_indices]
        else:
            sampled_values = None

        if size > n_nz_tokens:
            sampled_tokens.extend([0] * (size - len(sampled_tokens)))
            if self.gt_type == 'counts':
                sampled_values.extend([0.0] * (size - len(sampled_values)))

        return sampled_tokens, sampled_values
         
    def _get_segment_seq(self, 
                         item: int,
                         segment: int,
                         segment_seq_len: int,
                         ) -> Tuple[list[int], list[int]]:
            """
            Get gene tokens and values for a given segment based on a sampling
            strategy.

            Parameters
            -----------
            item:
                Index of the cell in the Hugging Face dataset.
            segment:
                Index of the segment for which tokens are retrieved.
            segment_seq_len:
                Desired length of the segment token sequence.

            Returns
            --------
            segment_tokens:
                List of tokens for a given segment.
            segment_values:
                List of values for a given segment.
            """
            # Only keep gene tokens and values in specified segment
            segment_start_idx = item['seg_tokens'].index(segment)
            if segment + 1 in item['seg_tokens']:
                segment_end_idx = item['seg_tokens'].index(segment+1)
            else:
                segment_end_idx = len(item['seg_tokens'])
            
            segment_tokens = item['gene_tokens'][
                segment_start_idx: segment_end_idx]
            if self.gt_type == 'counts':
                segment_values = item['gene_expr'][
                    segment_start_idx: segment_end_idx]
            else:
                segment_values = None

            # Validate that segment sequence length is specified correctly
            if (self.sampling_strategy is not None and 'rep' in
            self.sampling_strategy):
                pass
            else:
                if segment_seq_len > len(segment_tokens):
                    raise ValueError(
                        'Sequence length for a given segment cannot be larger '
                        'than segment size when not sampling with replacement.')

            # If no sampling strategy is specified, use all tokens up to
            # specified length
            if self.sampling_strategy is None:
                segment_tokens = segment_tokens[:segment_seq_len]
                if self.gt_type == 'counts':
                    segment_values = segment_values[:segment_seq_len]
            # Otherwise, sample a subset of tokens based on the sampling
            # strategy
            else:
                segment_n_nz_tokens = sum(
                    1 for token in segment_tokens if token != 0)

                segment_tokens, segment_values = self._sample_seq(
                    tokens=segment_tokens,
                    values=segment_values,
                    n_nz_tokens=segment_n_nz_tokens,
                    size=segment_seq_len)       
                    
            return segment_tokens, segment_values


class CellGraphDataset(CellBaseDataset):
    def __init__(self,
                 **base_dataset_kwargs,
                 ):
        """
        Torch CellGraphDataset class.

        Parameters
        -----------
        **base_dataset_kwargs:
            Keyword arguments for the initialization of the CellBaseDataset.
        """
        super().__init__(**base_dataset_kwargs)

    def __getitem__(self,
                    item: int,
                    ) -> Tuple[torch.Tensor,
                               torch.Tensor,
                               torch.Tensor,
                               list[int]]:
        # Retrieve Hugging Face item once
        item = self.dataset[item]

        # Get (sampled) gene tokens and values/positions for index cell segment
        tokens, values = self._get_segment_seq(
            item=item,
            segment=self.max_special_tokens, # first cell (index cell) segment
            segment_seq_len=self.seq_len_cell)
        if self.gt_type == 'rank':
            positions = [position if tokens[i] != 0 else 0 for i, position in 
                enumerate(list(range(1, len(tokens) + 1)))] 
        elif self.gt_type == 'counts':
            positions = None

        # Get non-padded segments for index cell segment
        segments = [
            self.max_special_tokens if token != 0 else 0 for token in tokens]

        # Get (sampled) gene tokens, values/positions and non-padded segments
        # for neighbor cell segments
        for segment in np.unique(item["seg_tokens"]):
            if segment > self.max_special_tokens: # neighbor cell segments
                segment_tokens, segment_values = self._get_segment_seq(
                    item=item,
                    segment=segment,
                    segment_seq_len=self.seq_len_cell)
                if self.gt_type == 'rank':   
                    positions.extend([position if segment_tokens[i] != 0 else 0
                        for i, position in enumerate(
                            list(range(1, len(segment_tokens) + 1)))])
                elif self.gt_type == 'counts':
                    values.extend(segment_values)
                tokens.extend(segment_tokens)
                segments.extend([segment if token != 0 else 0 for
                    token in segment_tokens])

        if len(tokens) > (self.seq_len_cell + self.seq_len_neighborhood):
            tokens = tokens[:self.seq_len_cell + self.seq_len_neighborhood]
            segments = segments[:self.seq_len_cell + self.seq_len_neighborhood]
            if self.gt_type == 'rank':
                positions = positions[
                    :self.seq_len_cell + self.seq_len_neighborhood]
            elif self.gt_type == 'counts':
                values = values[:self.seq_len_cell + self.seq_len_neighborhood]

        elif len(tokens) < (self.seq_len_cell + self.seq_len_neighborhood):
            tokens += [0] * (
                (self.seq_len_cell + self.seq_len_neighborhood) - len(tokens))
            segments += [0] * (
                (self.seq_len_cell + self.seq_len_neighborhood) - len(segments))
            if self.gt_type == 'rank':
                positions += [0] * (
                    (self.seq_len_cell + self.seq_len_neighborhood) -
                    len(positions))
            elif self.gt_type == 'counts':
                values += [0.0] * (
                    (self.seq_len_cell + self.seq_len_neighborhood) -
                    len(values))

        # Add special tokens
        if self.gt_type == 'rank':
            tokens, segments, positions = self._add_special_seq(
                item=item,
                tokens=tokens,
                segments=segments,
                positions=positions)            
        elif self.gt_type == 'counts':
            tokens, segments, values = self._add_special_seq(
                item=item,
                tokens=tokens,
                segments=segments,
                values=values)

        tokens = torch.tensor(tokens)
        segments = torch.tensor(segments)
        if self.gt_type == 'rank':
            positions = torch.tensor(positions)
            return tokens, segments, positions, item["cell_id"]
        elif self.gt_type == 'counts':
            values = torch.tensor(values)
            return tokens, segments, values, item["cell_id"]


class CellNeighborhoodDataset(CellBaseDataset):
    def __init__(self,
                 **base_dataset_kwargs
                 ):
        """
        Torch CellNeighborhoodDataset class.

        Parameters
        -----------
        **base_dataset_kwargs:
            Keyword arguments for the initialization of the CellBaseDataset.
        """
        super().__init__(**base_dataset_kwargs)

    def __getitem__(self,
                    item: int
                    ) -> Tuple[torch.Tensor,
                               torch.Tensor,
                               torch.Tensor,
                               list[int]]:
        # Retrieve Hugging Face item once
        item = self.dataset[item]

        # Get (sampled) gene tokens, values/positions and non-padded segments
        gene_tokens_cell, values_cell = self._get_segment_seq(
            item=item,
            segment=self.max_special_tokens, # cell seg
            segment_seq_len=self.seq_len_cell)
        gene_tokens_neigh, values_neigh = self._get_segment_seq(
            item=item,
            segment=self.max_special_tokens + 1, # neigh seg
            segment_seq_len=self.seq_len_neighborhood)
        tokens = gene_tokens_cell + gene_tokens_neigh
        segments = [
            self.max_special_tokens if gene_token != 0 else 0 for gene_token
            in gene_tokens_cell
            ] + [
            self.max_special_tokens + 1 if gene_token != 0 else 0 for
            gene_token in gene_tokens_neigh]
        if self.gt_type == 'rank':
            positions = list(range(1, len(gene_tokens_cell) + 1)) + list(
                range(1, len(gene_tokens_neigh) + 1))
            positions = [position if tokens[i] != 0 else 0 for i, position in 
                         enumerate(positions)]
        elif self.gt_type == 'counts':
            values = values_cell + values_neigh

        # Add special tokens
        if self.gt_type == 'rank':
            tokens, segments, positions = self._add_special_seq(
                item=item,
                tokens=tokens,
                segments=segments,
                positions=positions)            
        elif self.gt_type == 'counts':
            tokens, segments, values = self._add_special_seq(
                item=item,
                tokens=tokens,
                segments=segments,
                values=values)

        tokens = torch.tensor(tokens)
        segments = torch.tensor(segments)
        if self.gt_type == 'rank':
            positions = torch.tensor(positions)
            return tokens, segments, positions, item["cell_id"]
        elif self.gt_type == 'counts':
            values = torch.tensor(values)
            return tokens, segments, values, item["cell_id"]


def make_cell_dataset(tokenizer_type: Literal['cell_graph',
                                              'cell_neigh'],
                      **cell_dataset_kwargs,
                      ) -> Union[CellGraphDataset,
                                 CellNeighborhoodDataset]:
    """
    Based on tokenizer type, return CellGraphDataset or CellNeighborhoodDataset.
    """
    if tokenizer_type == 'cell_graph':
        cell_dataset = CellGraphDataset(**cell_dataset_kwargs)
    elif tokenizer_type  == 'cell_neigh':
        cell_dataset = CellNeighborhoodDataset(**cell_dataset_kwargs)

    return cell_dataset
