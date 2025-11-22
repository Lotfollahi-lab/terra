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
                 max_cls_tokens: int,
                 max_special_tokens: int,
                 tokenizer_type: Literal['cell_neighborhood', 'cell_graph'],
                 gt_type: Literal['rank', 'counts'],
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
            Hugging Face dataset with sequences of gene tokens and special
            tokens.
        vocab_size:
            Size of the vocabulary.
        seq_len_cell:
            Sequence length of the (index) cell tokens.
        seq_len_neighborhood:
            Sequence length of the neighborhood tokens.
        max_cls_tokens:
        max_special_tokens:
        tokenizer_type;
        gt_type:
            Gene transformer type.
        special_tokens:
            Special tokens to be included in the sequence processed by the
            model.
        sampling_strategy:
            Token sampling strategy.
        """
        if gt_type not in ['rank', 'counts']:
            raise ValueError(f'Invalid "gt_type": {gt_type}.')

        self.dataset = dataset
        self.len = len(self.dataset)
        self.n_nonzero_tokens = self.dataset['n_nonzero_tokens']
        self.vocab_size = vocab_size
        self.seq_len_cell = seq_len_cell
        self.seq_len_neighborhood = seq_len_neighborhood
        self.max_special_tokens = max_special_tokens
        self.gt_type = gt_type
        self.n_special_tokens = len(special_tokens)
        self.seq_len = (seq_len_cell +
                        seq_len_neighborhood +
                        self.n_special_tokens)
        self.special_tokens = special_tokens
        self.sampling_strategy = sampling_strategy

    def __len__(self) -> int:
        return self.len

    def _add_special_tokens_to_seq(self,
                                   tokens: List,
                                   segments: List,
                                   positions: List,
                                   gene_expr: List,
                                   item: int,
                                   ) -> Tuple[List[int], List[int]]:
        """
        Add special tokens to sequence and update segment and positions tokens.

        Parameters
        -----------
        tokens:
            Token sequence including all segments.
        segments:
            Segment tokens including all segments.
        positions:
            Position tokens including all segments.
        gene_expr:
            Gene expression values including all segments.
        item:
            Index of the cell in the Hugging Face dataset.

        Returns
        -----------
        tokens:
            Sequence of tokens with special tokens included at sequence start.
        segments:
            Segment labels with 0s for special tokens at sequence start.
        """
        if 'batch' in self.special_tokens:
            if self.gt_type == 'rank':
                tokens = item["batch_value_token"] + tokens
            elif self.gt_type == 'counts':
                tokens = item["batch_token"] + tokens
            gene_expr = item["batch_value"] + gene_expr
        if 'gene_panel' in self.special_tokens:
            if self.gt_type == 'rank':
                tokens = item["gene_panel_value_token"] + tokens
            elif self.gt_type == 'counts':
                tokens = item["gene_panel_token"] + tokens
            gene_expr = item["gene_panel_value"] + gene_expr
        if 'tissue' in self.special_tokens:
            if self.gt_type == 'rank':
                tokens = item["tissue_value_token"] + tokens
            elif self.gt_type == 'counts':
                tokens = item["tissue_token"] + tokens
            gene_expr = item["tissue_value"] + gene_expr
        if 'species' in self.special_tokens:
            if self.gt_type == 'rank':
                tokens = item["species_value_token"] + tokens
            elif self.gt_type == 'counts':
                tokens = item["species_token"] + tokens
            gene_expr = item["species_value"] + gene_expr
        if 'assay' in self.special_tokens:
            if self.gt_type == 'rank':
                tokens = item["assay_value_token"] + tokens
            elif self.gt_type == 'counts':
                tokens = item["assay_token"] + tokens
            gene_expr = item["assay_value"] + gene_expr
            
        n_cls_tokens = len(item["cls_tokens"])
        n_nz_cls_tokens = sum(
            1 for token in item["cls_tokens"] if token != 0)
        n_zero_cls_tokens = n_cls_tokens - n_nz_cls_tokens
        tokens = item["cls_tokens"] + tokens
        gene_expr = list(
            range(2, 2 + n_nz_cls_tokens)) + [0] * n_zero_cls_tokens + gene_expr

        segments = list(
            range(1, 1 + n_nz_cls_tokens)) + [0] * n_zero_cls_tokens + list(
            range(1 + n_nz_cls_tokens,  1 + n_nz_cls_tokens + (self.n_special_tokens - n_cls_tokens))) + segments
        positions = list(
            range(1, 1 + n_nz_cls_tokens)) + [0] * n_zero_cls_tokens + list(
            range(1 + n_nz_cls_tokens, 1 + n_nz_cls_tokens + (self.n_special_tokens - n_cls_tokens))) + positions

        return tokens, segments, positions, gene_expr

    def _create_sampled_token_and_count_seq(self,
                                            tokens: List,
                                            counts: List,
                                            n_nonzero_tokens: int,
                                            size: int,
                                            ) -> List[int]:
        """
        Sample a subset of tokens based on a sampling strategy.

        Parameters
        -----------
        tokens:
            List of tokens.
        counts:
            List of counts.
        n_nonzero_tokens:
            Number of nonzero tokens in `tokens`.
        size:
            Size of the sampled subset.
            
        Returns
        --------
        sampled_tokens:
            List of sampled tokens.
        sampled_counts:
            List of (corresponding) sampled counts.
        """
        if 'norm_count_rank_sampling' in self.sampling_strategy:
            # Calculate weights based on rank and number of nonzero tokens:
            # the higher the rank, the higher the weight
            # seq = [4, 1, 3, 2, 5, 0, 0, 0]
            # n_nonzero_tokens = 5  
            # sum_rank = 5 * (5 + 1) / 2.0 = 15.0
            # weights = [(n_nonzero_tokens - i)/sum_rank for i in range(
            #     n_nonzero_tokens)] 
            # = [0.333, 0.266, 0.2, 0.133, 0.066]
            # np.sum(weights) = 1.0
            sum_rank = (
                n_nonzero_tokens * (n_nonzero_tokens + 1) / 2.0) + 1e-9
            weights = [(n_nonzero_tokens - i)/sum_rank for i in range(
                n_nonzero_tokens)]
            assert np.isclose(np.sum(weights), 1.0)
        elif 'rand_sampling' in self.sampling_strategy:
            weights = np.ones(n_nonzero_tokens) / n_nonzero_tokens
        else:
            raise ValueError(f"'{self.sampling_strategy}' is invalid.")
            
        # Sample token indices based on weights
        sampled_indices = np.random.choice(
            np.arange(n_nonzero_tokens),
            size=min(size, n_nonzero_tokens),
            p=weights,
            replace=(True if 'rep' in self.sampling_strategy else False))
            
        # Sort sampled indices to preserve rank order
        sampled_indices = np.sort(sampled_indices)
        sampled_tokens = [tokens[i] for i in sampled_indices]
        sampled_counts = [counts[i] for i in sampled_indices]

        if size > n_nonzero_tokens:
            sampled_tokens.extend([0] * (size - len(sampled_tokens)))
            sampled_counts.extend([0.0] * (size - len(sampled_counts)))

        return sampled_tokens, sampled_counts
         
    def _get_gene_tokens_and_counts_for_segment(self, 
                                                item: int,
                                                segment: int,
                                                segment_seq_len: int,
                                                ) -> List[int]:
            """
            Get gene tokens and counts for a given segment based on a sampling
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
            segment_gene_tokens:
                List of tokens for a given segment.
            """
            # Only keep gene tokens and gene expr in specified segment
            segment_start_idx = item["seg_tokens"].index(segment)
            if segment + 1 in item["seg_tokens"]:
                segment_end_idx = item["seg_tokens"].index(segment+1)
            else:
                segment_end_idx = len(item["seg_tokens"])
            
            segment_gene_tokens = item["gene_tokens"][
                segment_start_idx: segment_end_idx]
            segment_gene_expr = item["gene_expr"][
                segment_start_idx: segment_end_idx]

            # Validate that segment sequence length is specified correctly
            if (self.sampling_strategy is not None and 'rep' in
            self.sampling_strategy):
                pass
            else:
                if segment_seq_len > len(segment_gene_tokens):
                    print(segment_seq_len)
                    print(len(segment_gene_tokens))
                    print(segment)
                    print(segment_gene_tokens)
                    print(item["seg_tokens"])
                    raise ValueError(
                        'Sequence length for a given segment cannot be larger '
                        'than segment size when not sampling with replacement.')

            # If no sampling strategy is specified, use all tokens up to
            # specified length
            if self.sampling_strategy is None:
                segment_gene_tokens = segment_gene_tokens[:segment_seq_len]
                segment_gene_expr = segment_gene_expr[:segment_seq_len]
            # Otherwise, sample a subset of tokens based on the sampling
            # strategy
            else:
                segment_n_nonzero_tokens = sum(
                    1 for token in segment_gene_tokens if token != 0)

                segment_gene_tokens, segment_gene_expr = self._create_sampled_token_and_count_seq(
                    tokens=segment_gene_tokens,
                    counts=segment_gene_expr,
                    n_nonzero_tokens=segment_n_nonzero_tokens,
                    size=segment_seq_len)       
                    
            return segment_gene_tokens, segment_gene_expr


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
                    item: int
                    ) -> Tuple[torch.Tensor,
                               torch.Tensor,
                               torch.Tensor,
                               List[int]]:
        # Retrieve Huggingface item once
        item = self.dataset[item]

        # Get (sampled) gene tokens and counts
        gene_tokens_cell, gene_expr_cell = self._get_gene_tokens_and_counts_for_segment(
            item=item,
            segment=self.max_special_tokens, # index cell seg
            segment_seq_len=self.seq_len_cell)
        segments = [self.max_special_tokens if gene_token != 0 else 0 for
                    gene_token in gene_tokens_cell]
        positions = list(range(1, len(gene_tokens_cell) + 1))
        gene_tokens_neighborhood = []
        gene_expr_neighborhood = []
        for segment in np.unique(item["seg_tokens"]):
            if segment > self.max_special_tokens: # neighbor cell segments
                segment_gene_tokens, segment_gene_expr = self._get_gene_tokens_and_counts_for_segment(
                    item=item,
                    segment=segment, # neighbor cell segs
                    segment_seq_len=self.seq_len_cell)
                gene_tokens_neighborhood.extend(segment_gene_tokens)
                gene_expr_neighborhood.extend(segment_gene_expr)
                segments.extend([segment if gene_token != 0 else 0 for
                                 gene_token in segment_gene_tokens])
                positions.extend(list(range(1, len(segment_gene_tokens) + 1)))
        tokens = gene_tokens_cell + gene_tokens_neighborhood
        positions = [position if tokens[i] != 0 else 0 for i, position in 
                     enumerate(positions)]
        gene_expr = gene_expr_cell + gene_expr_neighborhood

        if len(tokens) > (self.seq_len_cell + self.seq_len_neighborhood):
            tokens = tokens[:self.seq_len_cell + self.seq_len_neighborhood]
            segments = segments[:self.seq_len_cell + self.seq_len_neighborhood]
            positions = positions[:self.seq_len_cell + self.seq_len_neighborhood]
            gene_expr = gene_expr[:self.seq_len_cell + self.seq_len_neighborhood]
        elif len(tokens) < (self.seq_len_cell + self.seq_len_neighborhood):
            tokens += [0] * ((self.seq_len_cell + self.seq_len_neighborhood) - len(tokens))
            segments += [0] * ((self.seq_len_cell + self.seq_len_neighborhood) - len(segments))
            positions += [0] * ((self.seq_len_cell + self.seq_len_neighborhood) - len(positions))
            gene_expr += [0.0] * ((self.seq_len_cell + self.seq_len_neighborhood) - len(gene_expr))

        # Add special tokens
        tokens, segments, positions, gene_expr = self._add_special_tokens_to_seq(
            tokens=tokens,
            segments=segments,
            positions=positions,
            gene_expr=gene_expr,
            item=item)

        tokens = torch.tensor(tokens)
        segments = torch.tensor(segments)
        positions = torch.tensor(positions)
        gene_expr = torch.tensor(gene_expr)

        return tokens, segments, positions, gene_expr, item["cell_id"]


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
                               List[int]]:
        # Retrieve Huggingface item once
        item = self.dataset[item]

        # Get (sampled) gene tokens and counts
        gene_tokens_cell, gene_expr_cell = self._get_gene_tokens_and_counts_for_segment(
            item=item,
            segment=self.max_special_tokens, # cell seg
            segment_seq_len=self.seq_len_cell)
        gene_tokens_neighborhood, gene_expr_neighborhood = self._get_gene_tokens_and_counts_for_segment(
            item=item,
            segment=self.max_special_tokens + 1, # neighborhood seg
            segment_seq_len=self.seq_len_neighborhood)

        tokens = gene_tokens_cell + gene_tokens_neighborhood
        gene_expr = gene_expr_cell + gene_expr_neighborhood
        segments = [
            self.max_special_tokens if gene_token != 0 else 0 for gene_token
            in gene_tokens_cell
            ] + [
            self.max_special_tokens + 1 if gene_token != 0 else 0 for
            gene_token in gene_tokens_neighborhood]
        positions = list(range(1, len(gene_tokens_cell) + 1)) + list(
            range(1, len(gene_tokens_neighborhood) + 1))
        positions = [position if tokens[i] != 0 else 0 for i, position in 
                     enumerate(positions)]

        # Add special tokens
        tokens, segments, positions, gene_expr = self._add_special_tokens_to_seq(
            tokens=tokens,
            segments=segments,
            positions=positions,
            gene_expr=gene_expr,
            item=item)

        tokens = torch.tensor(tokens)
        segments = torch.tensor(segments)
        positions = torch.tensor(positions)
        gene_expr = torch.tensor(gene_expr)

        return tokens, segments, positions, gene_expr, item["cell_id"]


def make_cell_dataset(tokenizer_type: Literal['cell_graph',
                                              'cell_neighborhood'],
                      **cell_dataset_kwargs
                      ) -> Union[CellGraphDataset,
                                 CellNeighborhoodDataset]:
    """
    Based on tokenizer type, return CellGraphDataset or CellNeighborhoodDataset.
    """
    if tokenizer_type == 'cell_graph':
        cell_dataset = CellGraphDataset(tokenizer_type=tokenizer_type,
                                        **cell_dataset_kwargs)
    elif tokenizer_type  == 'cell_neighborhood':
        cell_dataset = CellNeighborhoodDataset(tokenizer_type=tokenizer_type,
                                               **cell_dataset_kwargs)

    return cell_dataset