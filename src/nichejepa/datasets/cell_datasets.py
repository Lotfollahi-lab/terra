from typing import Literal

import datasets
import numpy as np
import torch
from torch.utils.data import Dataset


class CellBaseDataset(Dataset):
    def __init__(self,
                 gt_type: Literal['rank', 'counts', 'combined'],
                 dataset: datasets.Dataset,
                 vocab_size: int,
                 seq_len_cell: int,
                 seq_len_neighborhood: int,
                 special_tokens: list[str] = [],
                 sampling_strategy: Literal['norm_value_rank_sampling',
                                            'norm_value_rank_sampling_rep',
                                            'rand_sampling',
                                            'rand_sampling_rep'] | None = None,
                 n_nonzero_tokens_list: list[int] | None = None,
                 sep_gene_tokens_neb: bool = False,
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
            Sequence length of the index cell (number of gene tokens).
        seq_len_neighborhood:
            Sequence length of the neighborhood (number of gene tokens).
        special_tokens:
            Special tokens to be included in the sequence.
        sampling_strategy:
            Token sampling strategy.
        n_nonzero_tokens_list:
            List of number of nonzero tokens.
        sep_gene_tokens_neb:
            If `True`, use separate tokens for genes in neighborhood vs
            to index cell.
        """
        if gt_type not in ['rank', 'counts', 'combined']:
            raise ValueError(f'Invalid "gt_type": {gt_type}.')
        
        self.gt_type = gt_type
        self.dataset = dataset
        self.len = len(self.dataset)
        self.vocab_size = vocab_size
        self.seq_len_cell = seq_len_cell
        self.seq_len_neighborhood = seq_len_neighborhood
        self.special_tokens = special_tokens
        self.n_special_tokens = len(special_tokens)
        self.seq_len = (seq_len_cell +
                        seq_len_neighborhood +
                        self.n_special_tokens)
        self.sampling_strategy = sampling_strategy
        if n_nonzero_tokens_list:
            self.n_nz_tokens = n_nonzero_tokens_list
        else:
            self.n_nz_tokens = self.dataset['n_nonzero_tokens']
        self.sep_gene_tokens_neb = sep_gene_tokens_neb

    def __len__(self) -> int:
        return self.len

    def _add_special_seq(self,
                         item: int,
                         item_dict: dict,
                         ) -> dict:
        """
        Add special tokens to sequence and update positions/values and
        segments.

        Parameters
        -----------
        item:
            Index of the cell in the Hugging Face dataset.
        item_dict:
            All attributes of the cell in the Hugging Face dataset,
            including positions, segments, tokens and values.

        Returns
        -----------
        item_dict:
            All attributes of the cell in the Hugging Face dataset with
            special tokens considered at sequence start.
        """
        for spc_tk in self.special_tokens:
            item_dict['tokens'] = item[f'{spc_tk}_token'] + item_dict['tokens']
            item_dict['values'] = item[f'{spc_tk}_value'] + item_dict['values']
            
        if self.gt_type != 'counts':
            # Add special token positions
            item_dict['positions'] = [0] * self.n_special_tokens + item_dict[
                'positions']

        # Add special token segments
        item_dict['segments'] = [0] * self.n_special_tokens + item_dict[
            'segments']

        return item_dict

    def _sample_seq(self,
                    tokens: list[int],
                    values: list[float] | None,
                    n_nz_tokens: int,
                    size: int,
                    ) -> tuple[list[int], list[float]]:
        """
        Sample a subset of gene tokens and corresponding values based on
        a sampling strategy.

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
            # Calculate weights based on rank and number of nonzero
            # tokens:
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
        sampled_values = [values[i] for i in sampled_indices]

        if size > n_nz_tokens:
            sampled_tokens.extend([0] * (size - len(sampled_tokens)))
            sampled_values.extend([0.0] * (size - len(sampled_values)))

        return sampled_tokens, sampled_values
         
    def _get_segment_seq(self, 
                         item: int,
                         segment: int,
                         segment_seq_len: int,
                         ) -> tuple[list[int], list[float]]:
            """
            Get gene tokens and values for a given segment based on a
            sampling strategy.

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
            # TODO: Fix tokenization after removal of 100 <cls> tokens
            seg_tokens = [
                (seg_token - 104) if seg_token != 0 else seg_token
                for seg_token in item['seg_tokens']]

            # Only keep gene tokens and values in specified segment
            segment_start_idx = seg_tokens.index(segment)
            if segment + 1 in seg_tokens:
                segment_end_idx = seg_tokens.index(segment+1)
            else:
                segment_end_idx = len(seg_tokens)
            
            segment_tokens = item['gene_tokens'][
                segment_start_idx: segment_end_idx]

            if segment != 1 and (self.sep_gene_tokens_neb):
                segment_tokens = [token + self.vocab_size if token != 0 else token for token in segment_tokens]

            segment_values = item['gene_expr'][segment_start_idx: segment_end_idx]

            # Validate that segment length is specified correctly
            if (self.sampling_strategy is not None and 'rep' in
            self.sampling_strategy):
                pass
            else:
                if segment_seq_len > len(segment_tokens):
                    raise ValueError(
                        'Sequence length for a given segment cannot be larger '
                        'than segment size when not sampling with replacement.'
                        )

            # If no sampling strategy is specified, use all tokens up to
            # specified length
            if self.sampling_strategy is None:
                segment_tokens = segment_tokens[:segment_seq_len]
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
            Keyword arguments for the initialization of the 
            CellBaseDataset.
        """
        super().__init__(**base_dataset_kwargs)

    def __getitem__(self,
                    item: int,
                    ) -> dict:
        item_dict = {}

        # Retrieve Hugging Face item once
        item = self.dataset[item]

        # Get (sampled) gene tokens, segments and values/positions for
        # index cell segment
        item_dict['tokens'], item_dict['values'] = self._get_segment_seq(
            item=item,
            segment=1, # index cell segment
            segment_seq_len=self.seq_len_cell)
        if self.gt_type != 'counts':
            item_dict['positions'] = [
                position if tokens[i] != 0 else 0 for i, position in 
                enumerate(list(range(1, len(tokens) + 1)))]
        else:
            item_dict['positions'] = None
        item_dict['segments'] = [1 if token != 0 else 0 for token in tokens]

        # Get (sampled) gene tokens, segments and values/positions for
        # neighbor cell segments
        # TODO: Fix tokenization index after removal of 100 <cls> tokens
        seg_tokens = [
            (seg_token - 104) if seg_token != 0 else seg_token for
            seg_token in item['seg_tokens']]

        for segment in np.unique(seg_tokens):
            if segment > 1: # neighbor cell segments
                segment_tokens, segment_values = self._get_segment_seq(
                    item=item,
                    segment=segment,
                    segment_seq_len=self.seq_len_cell)
                if self.gt_type != 'counts':
                    item_dict['positions'].extend(
                        [position if segment_tokens[i] != 0 else 0
                        for i, position in enumerate(
                            list(range(1, len(segment_tokens) + 1)))])
                elif self.gt_type != 'rank':
                    item_dict['values'].extend(segment_values)
                item_dict['tokens'].extend(segment_tokens)
                item_dict['segments'].extend([segment if token != 0 else 0 for
                    token in segment_tokens])

        if len(tokens) > (self.seq_len_cell + self.seq_len_neighborhood):
            item_dict['tokens'] = item_dict['tokens'][
                :self.seq_len_cell + self.seq_len_neighborhood]
            item_dict['segments'] = item_dict['segments'][
                :self.seq_len_cell + self.seq_len_neighborhood]
            if self.gt_type != 'counts':
                item_dict['positions'] = item_dict['positions'][
                    :self.seq_len_cell + self.seq_len_neighborhood]
            elif self.gt_type != 'rank':
                item_dict['values'] = item_dict['values'][
                    :self.seq_len_cell + self.seq_len_neighborhood]

        elif len(tokens) < (self.seq_len_cell + self.seq_len_neighborhood):
            item_dict['tokens'] += [0] * (
                (self.seq_len_cell + self.seq_len_neighborhood) - len(tokens))
            item_dict['segments'] += [0] * (
                (self.seq_len_cell + self.seq_len_neighborhood) - len(segments)
                )
            if self.gt_type != 'counts':
                item_dict['positions'] += [0] * (
                    (self.seq_len_cell + self.seq_len_neighborhood) -
                    len(positions))
            elif self.gt_type != 'rank':
                item_dict['values'] += [0.0] * (
                    (self.seq_len_cell + self.seq_len_neighborhood) -
                    len(values))

        # Add special tokens
        item_dict = self._add_special_seq(item=item,
                                          item_dict=item_dict)

        for key in item_dict.keys():
            item_dict[key] = torch.tensor(item_dict[key])

        item_dict['cell_id'] = item['cell_id']
        return item_dict


class CellNeighborhoodDataset(CellBaseDataset):
    def __init__(self,
                 **base_dataset_kwargs
                 ):
        """
        Torch CellNeighborhoodDataset class.

        Parameters
        -----------
        **base_dataset_kwargs:
            Keyword arguments for the initialization of the
            CellBaseDataset.
        """
        super().__init__(**base_dataset_kwargs)

    def __getitem__(self,
                    item: int
                    ) -> tuple[torch.Tensor,
                               torch.Tensor,
                               torch.Tensor,
                               list[int]]:
        # Retrieve Hugging Face item once
        item = self.dataset[item]

        # Get (sampled) gene tokens, values/positions and non-padded
        # segments
        gene_tokens_cell, values_cell = self._get_segment_seq(
            item=item,
            segment=1, # cell seg
            segment_seq_len=self.seq_len_cell)
        gene_tokens_neigh, values_neigh = self._get_segment_seq(
            item=item,
            segment=2, # neigh seg
            segment_seq_len=self.seq_len_neighborhood)
        tokens = gene_tokens_cell + gene_tokens_neigh
        segments = [
            1 if gene_token != 0 else 0 for gene_token in gene_tokens_cell] + [
            2 if gene_token != 0 else 0 for gene_token in gene_tokens_neigh]
        positions = list(range(1, len(gene_tokens_cell) + 1)) + list(
            range(1, len(gene_tokens_neigh) + 1))
        positions = [position if tokens[i] != 0 else 0 for i, position in 
                        enumerate(positions)]
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


def init_cell_dataset(tokenizer_type: Literal['cell_graph',
                                              'cell_neigh'],
                      **cell_dataset_kwargs,
                      ) -> CellGraphDataset | CellNeighborhoodDataset:
    """
    Initialize CellDataset based on tokenizer type.
    """
    if tokenizer_type == 'cell_graph':
        cell_dataset = CellGraphDataset(**cell_dataset_kwargs)
    elif tokenizer_type  == 'cell_neigh':
        cell_dataset = CellNeighborhoodDataset(**cell_dataset_kwargs)

    return cell_dataset
