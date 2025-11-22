from typing import Literal

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CellBaseDataset(Dataset):
    def __init__(self,
                 gt_type: Literal['rank', 'counts', 'combined'],
                 cell_pos_enc: Literal['segment', 'coord'],
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
                 include_cell_id: bool = False,
                 sep_gene_tokens_neb: bool = False,
                 ):
        """
        Torch CellBaseDataset class.

        Parameters
        -----------
        gt_type:
            Gene transformer type.
        cell_pos_enc:
            Encoding used to encode cell positions.
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
        include_cell_id:
            If `True`, return cell ID string in getitem().
        sep_gene_tokens_neb:
            If `True`, use separate tokens for genes in neighborhood vs
            index cell.
        """
        if gt_type not in ['rank', 'counts', 'combined']:
            raise ValueError(f'Invalid "gt_type": {gt_type}.')
        if cell_pos_enc not in ['segment', 'coord']:
            raise ValueError(f'Invalid "cell_pos_enc": {cell_pos_enc}.')
        
        self.gt_type = gt_type
        self.cell_pos_enc = cell_pos_enc
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
        self.n_segments = (seq_len_cell + seq_len_neighborhood) / seq_len_cell
        self.sampling_strategy = sampling_strategy
        if n_nonzero_tokens_list:
            self.n_nz_tokens = n_nonzero_tokens_list
        else:
            self.n_nz_tokens = list(self.dataset['n_nonzero_tokens'])
        self.include_cell_id = include_cell_id
        self.sep_gene_tokens_neb = sep_gene_tokens_neb

    def __len__(self) -> int:
        return self.len

    def _add_special_seq(self,
                         item: int,
                         item_dict: dict,
                         ) -> dict:
        """
        Add special tokens to sequence and update positions, segments,
        and values.

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
        # Add <cls> token
        item_dict['tokens'] = torch.cat([2, item_dict['tokens']])

        for spc_tk in self.special_tokens:
            item_dict['tokens'] = torch.cat(
                [item[f'{spc_tk}_token'],
                 item_dict['tokens']])

            if self.gt_type != 'rank':
                # Add <cls> token
                item_dict['values'] = torch.cat(
                    [0, item_dict['values']])
                item_dict['values'] = torch.cat(
                    [item[f'{spc_tk}_value'],
                     item_dict['values']])
            
        if self.gt_type != 'counts':
            # Add <cls> and special token positions
            item_dict['positions'] = torch.cat(
                [torch.zeros(self.n_special_tokens + 1, dtype=torch.long),
                 item_dict['positions']])

        # Add <cls> and special token segments
        item_dict['segments'] = torch.cat(
            [torch.zeros(self.n_special_tokens + 1, dtype=torch.long),
             item_dict['segments']])

        # Add <cls> and special token coords
        if self.cell_pos_enc == 'coord':
            item_dict['rel_x_coords'] = torch.cat(
                [torch.full((self.n_special_tokens + 1,),
                 float('-inf'), dtype=torch.float),
                 item_dict['rel_x_coords']])   
            item_dict['rel_y_coords'] = torch.cat(
                [torch.full((self.n_special_tokens + 1,),
                 float('-inf'), dtype=torch.float),
                 item_dict['rel_y_coords']])

        return item_dict

    def _sample_seq(self,
                    tokens: list[int],
                    values: list[float] | None,
                    rel_x_coords: list[float] | None,
                    rel_y_coords: list[float] | None,
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
        rel_x_coords:
            List of relative x coordinates.
        rel_y_coords:
            List of relative y coordinates.
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
        if values is not None:
            sampled_values = [values[i] for i in sampled_indices]
        else:
            sampled_values = None
        if rel_x_coords is not None: # the coordinates are all the same so sampling is just for length
            sampled_rel_x_coords = rel_x_coords[:len(sampled_indices)]
        else:
            sampled_rel_x_coords = None
        if rel_y_coords is not None: # the coordinates are all the same so sampling is just for length
            sampled_rel_y_coords = rel_y_coords[:len(sampled_indices)]
        else:
            sampled_rel_y_coords = None

        if size > n_nz_tokens:
            sampled_tokens.extend([0] * (size - len(sampled_tokens)))
            if values is not None:
                sampled_values.extend([0.0] * (size - len(sampled_values)))
            if sampled_rel_x_coords is not None:
                sampled_rel_x_coords.extend([float('-inf')] * (
                    size - len(sampled_rel_x_coords)))
            if sampled_rel_y_coords is not None:
                sampled_rel_y_coords.extend([float('-inf')] * (
                    size - len(sampled_rel_y_coords)))

        return (
            sampled_tokens,
            sampled_values,
            sampled_rel_x_coords,
            sampled_rel_y_coords)
         
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
            # TODO: Fix tokenization index after removal of 100 <cls> tokens
            #seg_tokens = torch.where(
            #    item['seg_tokens'] != 0,
            #    item['seg_tokens'] - 104,
            #    item['seg_tokens'])

            # Only keep gene tokens, values, and coords of specified
            # segment
            curr_seg_mask = item['seg_tokens'] == segment
            next_seg_mask = (item['seg_tokens'] == (segment + 1))
            segment_start_idx = torch.nonzero(
                curr_seg_mask, as_tuple=True)[0][0]
            if next_seg_mask.any():
                segment_end_idx = torch.nonzero(
                    next_seg_mask, as_tuple=True)[0][0]
            else:
                segment_end_idx = item['seg_tokens'].size(0)
            segment_tokens = item['gene_tokens'][
                segment_start_idx: segment_end_idx]
            if self.gt_type != 'rank':
                segment_values = item['gene_expr'][
                    segment_start_idx: segment_end_idx]
            else:
                segment_values = None
            if self.cell_pos_enc == 'coord':
                segment_rel_x_coords = item['rel_x_coord'][
                    segment_start_idx: segment_end_idx]
                segment_rel_y_coords = item['rel_y_coord'][
                    segment_start_idx: segment_end_idx]
            else:
                segment_rel_x_coords = None
                segment_rel_y_coords = None

            if segment != 1 and self.sep_gene_tokens_neb:
                # Create new tokens for neighbor genes
                segment_token_nz_mask = segment_tokens.ne(0)
                segment_tokens[segment_token_nz_mask] += self.vocab_size

            # Validate that segment length is specified correctly
            if (self.sampling_strategy is not None and 'rep' in
            self.sampling_strategy):
                pass
            else:
                if segment_tokens.size(0) < segment_seq_len:
                    raise ValueError(
                        'Sequence length for a given segment cannot be larger '
                        'than segment size when not sampling with replacement.'
                        )

            # If no sampling strategy is specified, use all tokens up to
            # specified length
            if self.sampling_strategy is None:
                segment_tokens = segment_tokens[:segment_seq_len]
                if self.gt_type != 'rank':
                    segment_values = segment_values[:segment_seq_len]
                if self.cell_pos_enc == 'coord':
                    segment_rel_x_coords = segment_rel_x_coords[
                        :segment_seq_len]
                    segment_rel_y_coords = segment_rel_y_coords[
                        :segment_seq_len]
            # Otherwise, sample a subset of tokens based on the sampling
            # strategy
            else:
                segment_n_nz_tokens = int(
                    torch.count_nonzero(segment_tokens))

                segment_tokens, \
                segment_values, \
                segment_rel_x_coords, \
                segment_rel_y_coords = self._sample_seq(
                    tokens=segment_tokens,
                    values=segment_values,
                    rel_x_coords=segment_rel_x_coords,
                    rel_y_coords=segment_rel_y_coords,
                    n_nz_tokens=segment_n_nz_tokens,
                    size=segment_seq_len)       
                    
            return (segment_tokens,
                    segment_values,
                    segment_rel_x_coords,
                    segment_rel_y_coords)


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

        # Expand spatial coordinates (TODO: if statement to support old API)
        if 'rel_x_coord' in item.keys():
            if len(item['rel_x_coord']) != len(item['gene_tokens']):
                item['rel_x_coord'] = torch.repeat_interleave(
                    item['rel_x_coord'], self.seq_len_cell)
                item['rel_y_coord'] = torch.repeat_interleave(
                    item['rel_y_coord'], self.seq_len_cell)

        # Add segment to item (TODO: if statement to support old API)
        if 'seg_tokens' not in item.keys():
            seg_tokens = torch.arange(1, self.n_segments + 1)
            seg_tokens = torch.repeat_interleave(
                seg_tokens, self.seq_len_cell)
            # Mask out positions where gene_tokens == 0
            seg_tokens = seg_tokens * (item['gene_tokens'] != 0).long()
            item['seg_tokens'] = seg_tokens

        # TODO: add special tokens from token dict directly (1 value per row)

        # Get (sampled) gene tokens, positions, segments, and values for
        # index cell segment
        item_dict['tokens'], \
        item_dict['values'], \
        item_dict['rel_x_coords'], \
        item_dict['rel_y_coords'] = self._get_segment_seq(
            item=item,
            segment=1, # index cell segment
            segment_seq_len=self.seq_len_cell)

        if self.gt_type == 'rank':
            del(item_dict['values'])

        segment_token_zero_mask = item_dict['tokens'].eq(0)
            
        if self.gt_type != 'counts':
            item_dict['positions'] = torch.arange(
                1, item_dict['tokens'].size(0) + 1, dtype=torch.long)
            item_dict['positions'][segment_token_zero_mask] = torch.tensor(
                0, dtype=torch.long)
        item_dict['segments'] = torch.ones_like(item_dict['tokens'])
        item_dict['segments'][segment_token_zero_mask] = torch.tensor(
            0, dtype=torch.long)

        if self.cell_pos_enc == 'coord':
            item_dict['rel_x_coords'][segment_token_zero_mask] = torch.tensor(
                float('-inf'), dtype=torch.float)
            item_dict['rel_y_coords'][segment_token_zero_mask] = torch.tensor(
                float('-inf'), dtype=torch.float)
        else:
            del(item_dict['rel_x_coords'])
            del(item_dict['rel_y_coords'])

        # Get (sampled) gene tokens, positions, segments and values for
        # neighbor cell segments
        # TODO: Fix tokenization index after removal of 100 <cls> tokens
        #seg_tokens = torch.where(
        #    item['seg_tokens'] != 0,
        #    item['seg_tokens'] - 104,
        #    item['seg_tokens'])

        for segment in torch.unique(item['seg_tokens']):
            if segment.item() > 1: # neighbor cell segments
                segment_tokens, \
                segment_values, \
                segment_rel_x_coords, \
                segment_rel_y_coords = self._get_segment_seq(
                    item=item,
                    segment=segment.item(),
                    segment_seq_len=self.seq_len_cell)

                segment_zero_mask = segment_tokens.eq(0)

                if self.gt_type != 'counts':
                    segment_pos = torch.arange(
                        1, segment_tokens.size(0) + 1, dtype=torch.long)
                    segment_pos[segment_zero_mask] = torch.tensor(
                        0, dtype=torch.long)
                    item_dict['positions'] = torch.cat(
                        [item_dict['positions'], segment_pos], dim=0)
                if self.gt_type != 'rank':
                    item_dict['values'] = torch.cat(
                        [item_dict['values'], segment_values], dim=0)
                item_dict['tokens'] = torch.cat(
                    [item_dict['tokens'], segment_tokens], dim=0)
                segment_tensor = torch.where(
                    segment_tokens != 0,
                    segment,
                    torch.tensor(0, dtype=torch.long)).to(dtype=torch.long)
                item_dict['segments'] = torch.cat(
                    [item_dict['segments'], segment_tensor], dim=0)
                if self.cell_pos_enc == 'coord':
                    segment_rel_x_coords[segment_zero_mask] = torch.tensor(
                        float('-inf'), dtype=torch.float)
                    segment_rel_y_coords[segment_zero_mask] = torch.tensor(
                        float('-inf'), dtype=torch.float)
                    item_dict['rel_x_coords'] = torch.cat(
                    [item_dict['rel_x_coords'], segment_rel_x_coords], dim=0)
                    item_dict['rel_y_coords'] = torch.cat(
                    [item_dict['rel_y_coords'], segment_rel_y_coords], dim=0)

        current_len = item_dict['tokens'].size(0)
        target_len = self.seq_len_cell + self.seq_len_neighborhood

        if current_len > target_len:
            # Truncate tokens
            item_dict['tokens'] = item_dict['tokens'][:target_len]
            item_dict['segments'] = item_dict['segments'][:target_len]
            if self.gt_type != 'counts':
                item_dict['positions'] = item_dict['positions'][:target_len]
            if self.gt_type != 'rank':
                item_dict['values'] = item_dict['values'][:target_len]
            if self.cell_pos_enc == 'coord':
                item_dict['rel_x_coords'] = item_dict['rel_x_coords'][
                    :target_len]
                item_dict['rel_y_coords'] = item_dict['rel_y_coords'][
                    :target_len]
        elif current_len < target_len:
            # Pad tokens
            pad_len = target_len - current_len
            item_dict['tokens'] = F.pad(
                item_dict['tokens'], (0, pad_len), value=0)
            item_dict['segments'] = F.pad(
                item_dict['segments'], (0, pad_len), value=0)
            if self.gt_type != 'counts':
                item_dict['positions'] = F.pad(
                    item_dict['positions'], (0, pad_len), value=0)
            if self.gt_type != 'rank':
                item_dict['values'] = F.pad(
                    item_dict['values'], (0, pad_len), value=0.0)
            if self.cell_pos_enc == 'coord':
                item_dict['rel_x_coords'] = F.pad(
                    item_dict['rel_x_coords'],
                    (0, pad_len),
                    value=float('-inf'))
                item_dict['rel_y_coords'] = F.pad(
                    item_dict['rel_y_coords'],
                    (0, pad_len),
                    value=float('-inf'))                     

        # Add special tokens
        if self.n_special_tokens > 0:
            item_dict = self._add_special_seq(item=item,
                                              item_dict=item_dict)

        # Add cell ID
        if self.include_cell_id:
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
                    ) -> dict:
        item_dict = {}

        # Retrieve Hugging Face item once
        item = self.dataset[item]
        
        # Get (sampled) gene tokens, positions, segments, and values
        gene_tokens_cell, \
        values_cell, \
        rel_x_coords_cell, \
        rel_y_coords_cell = self._get_segment_seq(
            item=item,
            segment=1, # cell seg
            segment_seq_len=self.seq_len_cell)
        gene_tokens_neigh, \
        values_neigh, \
        rel_x_coords_neigh, \
        rel_y_coords_neigh = self._get_segment_seq(
            item=item,
            segment=2, # neigh seg
            segment_seq_len=self.seq_len_neighborhood)
        item_dict['tokens'] = torch.cat(
            [gene_tokens_cell, gene_tokens_neigh], dim=0)

        segments_cell = torch.where(
            gene_tokens_cell != 0, torch.tensor(1), torch.tensor(0)).to(
                dtype=torch.long)
        segments_neigh = torch.where(
            gene_tokens_neigh != 0, torch.tensor(2), torch.tensor(0)).to(
                dtype=torch.long)
        item_dict['segments'] = torch.cat(
            [segments_cell, segments_neigh], dim=0)
        if self.cell_pos_enc == 'coord':
            item_dict['rel_x_coords'] = torch.cat(
                [rel_x_coords_cell, rel_x_coords_neigh], dim=0)
            item_dict['rel_y_coords'] = torch.cat(
                [rel_y_coords_cell, rel_y_coords_neigh], dim=0)

        if self.gt_type != 'count':
            item_dict['positions'] = torch.cat([
                torch.arange(1, gene_tokens_cell.size(0) + 1),
                torch.arange(1, gene_tokens_neigh.size(0) + 1)])
            item_dict['positions'] = item_dict['positions'] * (
                item_dict['tokens'] != 0).to(positions.dtype)

        if self.gt_type != 'rank':
            item_dict['values'] = torch.cat([values_cell, values_neigh], dim=0)

        # Add special tokens
        item_dict = self._add_special_seq(item=item,
                                          item_dict=item_dict)

        # Add cell ID
        if self.include_cell_id:
            item_dict['cell_id'] = item['cell_id']
        
        return item_dict


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
