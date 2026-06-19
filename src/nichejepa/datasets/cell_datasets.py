from typing import Literal

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# Encoder modes that need per-token rel_x / rel_y coordinates attached
# to each batch item. 'coord' uses them directly via sincos; 'polar'
# recomputes (log(1+r), theta) from them at encoder forward time;
# 'alibi' uses them to build the per-head attention distance bias;
# 'polar+alibi' uses both; 'laplacian' builds a spatial graph from
# them and uses its Laplacian eigenvectors as per-cell PE; 'rope'
# uses them to rotate q/k inside attention.
# 'segment' and 'none' only need the segment IDs (or nothing), so
# coords are not attached.
_COORD_BASED_POS_ENCS = (
    'coord', 'polar', 'alibi', 'polar+alibi', 'laplacian', 'rope')


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
                 nz_spc: bool = True,
                 pad_special_tokens: bool = False,
                 gene_panel_subsample: bool = False,
                 gene_panel_subsample_min_ratio: float = 0.25,
                 gene_panel_subsample_max_ratio: float = 1.0,
                 panel_size_norm: float = 1.0,
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
        # 'polar' and 'alibi' both consume the same per-cell rel_x /
        # rel_y the dataset already provides in 'coord' mode (polar
        # recomputes (r, theta) from them at the encoder; alibi uses
        # them to build the attention distance bias). 'segment' uses
        # only segment IDs. So at the dataset level, polar/alibi need
        # exactly the same rel-coord columns as coord.
        if cell_pos_enc not in [
                'none', 'segment', 'coord', 'polar', 'alibi',
                'polar+alibi', 'laplacian', 'rope']:
            raise ValueError(f'Invalid "cell_pos_enc": {cell_pos_enc}.')
        
        self.gt_type = gt_type
        self.cell_pos_enc = cell_pos_enc

        exclude_cols = [
            #'gene_panel_value',
            #'assay_value',
            'species_value',
            #'tissue_value'
            ]
        #if self.cell_pos_enc != 'coord':
        #    exclude_cols += [
        #        'rel_x_coord',
        #        'rel_y_coord']
        #if not include_cell_id:
        #    exclude_cols += ['cell_id']
        for col in exclude_cols:
            if col in dataset.features.keys():
                dataset = dataset.remove_columns(col)

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
        self.nz_spc = nz_spc
        self.pad_special_tokens = pad_special_tokens

        # Gene-panel subsampling (training augmentation) + panel-size
        # conditioning. ``gene_panel_active`` is driven purely by whether the
        # 'gene_panel' special token is configured, so the conditioning scalar
        # (item_dict['panel_size']) is produced at BOTH training and inference;
        # ``gene_panel_subsample`` additionally drops genes (training only).
        self.gene_panel_active = 'gene_panel' in self.special_tokens
        self.gene_panel_subsample = gene_panel_subsample
        self.gene_panel_subsample_min_ratio = gene_panel_subsample_min_ratio
        self.gene_panel_subsample_max_ratio = gene_panel_subsample_max_ratio
        # Divisor applied to the (effective) panel size before it is embedded,
        # to keep the MLP input ~O(1). Set to a typical full panel size for the
        # corpus (e.g. 1000). Default 1.0 = raw count.
        self.panel_size_norm = panel_size_norm

        # Auto-detect per-cell metadata columns (anything ending in
        # ``_value`` in the HF dataset features). These are exposed
        # as scalar fields in the per-cell item_dict independently of
        # ``special_tokens``, so downstream batch-correction code can
        # read batch / assay / etc. labels even when nothing is
        # concatenated into the encoder's input sequence. The
        # encoder-side ``special_tokens`` config is now strictly
        # about what gets INTO the token stream; this metadata path
        # is what gets used by AdaLN / adv_classifier /
        # distribution_alignment / cycle_consistency /
        # special_token_moe for their batch-label needs.
        # Only include numeric ``_value`` columns. HF dataset
        # features expose a dtype via ``.dtype`` for primitive columns
        # and via the ``feature.dtype`` of a Sequence's inner element
        # for list-of-X columns. A string-typed metadata column (e.g.
        # if someone names a column ``tissue_value`` storing strings)
        # would crash ``torch.tensor(...)`` in ``_attach_metadata`` --
        # filter it out at init time so per-item iteration stays
        # fast and crash-free.
        def _is_numeric_feature(feat) -> bool:
            try:
                # Sequence-of-X: inspect the element dtype.
                inner = getattr(feat, 'feature', None)
                dt = getattr(inner if inner is not None else feat,
                             'dtype', None)
                if dt is None:
                    return False
                dt = str(dt)
                return any(prefix in dt for prefix in (
                    'int', 'float', 'bool'))
            except Exception:
                return False
        self.metadata_keys = [
            col for col in self.dataset.features.keys()
            if col.endswith('_value')
            and _is_numeric_feature(self.dataset.features[col])
        ]

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
        for spc_tk in self.special_tokens:
            if self.gt_type != 'rank':
                item_dict['tokens'] = torch.cat(
                    [item[f'{spc_tk}_token'],
                    item_dict['tokens']])
            else:
                if self.vocab_size == 2785:
                    spv_idx_subtract = torch.tensor([1674]) # mus musculus token dict
                else:
                    spv_idx_subtract = torch.tensor([21957]) # homo sapiens token dict
                item_dict['tokens'] = torch.cat(
                    [item[f'{spc_tk}_value'] + spv_idx_subtract, # see tokenizers module
                    item_dict['tokens']])

            if self.gt_type != 'rank':
                item_dict['values'] = torch.cat(
                    [item[f'{spc_tk}_value'],
                     item_dict['values']])
            
        if self.gt_type != 'counts':
            if self.nz_spc:
                # Add special token positions
                item_dict['positions'] = torch.cat(
                    [torch.arange(
                        1,
                        self.n_special_tokens + 1,
                        dtype=torch.long),
                    item_dict['positions']])
            else:
                # Add special token positions
                item_dict['positions'] = torch.cat(
                    [torch.zeros(self.n_special_tokens, dtype=torch.long),
                    item_dict['positions']])

        # Add special token segments
        if self.nz_spc:
            item_dict['segments'] = torch.cat(
                [torch.arange(
                    1,
                    self.n_special_tokens + 1,
                    dtype=torch.long),
                item_dict['segments']])
        else:
            item_dict['segments'] = torch.cat(
                [torch.zeros(self.n_special_tokens, dtype=torch.long),
                item_dict['segments']])

        # Add special token coords
        if self.cell_pos_enc in _COORD_BASED_POS_ENCS:
            item_dict['rel_x_coords'] = torch.cat(
                [torch.full((self.n_special_tokens,),
                 float('-inf'), dtype=torch.float),
                 item_dict['rel_x_coords']])   
            item_dict['rel_y_coords'] = torch.cat(
                [torch.full((self.n_special_tokens,),
                 float('-inf'), dtype=torch.float),
                 item_dict['rel_y_coords']])

        return item_dict

    def _attach_metadata(self,
                         item: dict,
                         item_dict: dict,
                         ) -> dict:
        """Expose per-cell metadata (``*_value`` columns from the HF
        dataset) as scalar fields in ``item_dict``. These are the
        offset-subtracted spv_*_<id> token IDs for batch / assay /
        gene_panel / tissue / ... -- one integer per cell.

        Importantly this happens REGARDLESS of ``self.special_tokens``,
        so batch_correction mechanisms downstream can read the labels
        even when no special tokens are concatenated into the
        sequence. The encoder-side input is decoupled from the
        loss-side label source.

        The collator stacks across cells; scalar 0-d tensors become
        ``(B,)`` LongTensors per metadata key. The encoder ignores
        these keys; batch_correction code reads them by name.
        """
        for key in self.metadata_keys:
            if key not in item:
                continue
            val = item[key]
            # HF dataset usually returns 1-element lists / tensors.
            # Normalize to a 0-d long tensor so the collator stacks
            # into a clean (B,) per key.
            if isinstance(val, list):
                val = val[0] if len(val) > 0 else 0
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val)
            val = val.reshape(-1)
            if val.numel() == 0:
                continue
            item_dict[key] = val[0].long()
        return item_dict

    def _apply_gene_panel(self,
                          item: dict,
                          item_dict: dict,
                          ) -> dict:
        """Gene-panel subsampling (training) + panel-size conditioning.

        No-op unless the 'gene_panel' special token is configured. When
        active, attaches a continuous, normalized panel-size scalar to
        ``item_dict['panel_size']`` (consumed by the model's panel_size_embed
        at the gene_panel slot) and a placeholder ``item['gene_panel_value']``
        for the in-sequence value channel (the model overrides that slot).

        When ``gene_panel_subsample`` is on, additionally drops the SAME random
        subset of genes across ALL segments (panel-consistent simulation of a
        smaller measured panel) and sets the conditioning value to the
        simulated effective panel size ``round(keep_ratio * native_panel)``.
        Must run AFTER the per-segment sequence is assembled/padded and BEFORE
        ``_add_special_seq``.
        """
        if not self.gene_panel_active:
            return item_dict

        # Native panel size for this cell's dataset (stored at tokenization).
        panel_full = item.get('panel_size')
        if panel_full is None:
            raise KeyError(
                "'gene_panel' is configured but the tokenized dataset has no "
                "'panel_size' column. Re-tokenize with the updated tokenizer.")
        if isinstance(panel_full, torch.Tensor):
            panel_full = panel_full.reshape(-1)[0].item()
        elif isinstance(panel_full, (list, tuple)):
            panel_full = panel_full[0]
        panel_full = float(panel_full)

        effective = panel_full
        if self.gene_panel_subsample:
            lo = self.gene_panel_subsample_min_ratio
            hi = self.gene_panel_subsample_max_ratio
            keep_ratio = lo + (hi - lo) * float(torch.rand(1).item())

            tokens = item_dict['tokens']
            nz = tokens != 0
            # Map neighbor-shifted tokens back to their base gene id so a
            # dropped gene is dropped consistently in the cell AND neighbor
            # segments (panel-consistent).
            if self.sep_gene_tokens_neb:
                base_ids = torch.where(
                    tokens >= self.vocab_size, tokens - self.vocab_size, tokens)
            else:
                base_ids = tokens
            unique_ids = torch.unique(base_ids[nz])
            n_unique = int(unique_ids.numel())
            if n_unique > 0:
                kept_ids = unique_ids[torch.rand(n_unique) < keep_ratio]
                drop_pos = nz & ~torch.isin(base_ids, kept_ids)
                if drop_pos.any():
                    item_dict['tokens'] = torch.where(
                        drop_pos, torch.zeros_like(tokens), tokens)
                    if 'values' in item_dict:
                        item_dict['values'] = item_dict['values'].masked_fill(
                            drop_pos, 0.0)
                    if 'positions' in item_dict:
                        item_dict['positions'] = item_dict[
                            'positions'].masked_fill(drop_pos, 0)
                    item_dict['segments'] = item_dict['segments'].masked_fill(
                        drop_pos, 0)
                    if self.cell_pos_enc in _COORD_BASED_POS_ENCS:
                        item_dict['rel_x_coords'] = item_dict[
                            'rel_x_coords'].masked_fill(drop_pos, float('-inf'))
                        item_dict['rel_y_coords'] = item_dict[
                            'rel_y_coords'].masked_fill(drop_pos, float('-inf'))
            effective = max(1.0, float(round(keep_ratio * panel_full)))

        item_dict['panel_size'] = torch.tensor(
            effective / self.panel_size_norm, dtype=torch.float32)
        # Ensure _add_special_seq has a gene_panel value to prepend even when
        # the tokenized dataset has no 'gene_panel_value' column (e.g.
        # include_special_tokens=False). Only set a placeholder when ABSENT, so
        # we do NOT clobber the real spv_gene_panel* metadata value that
        # _attach_metadata exposes for downstream batch-correction routing. The
        # model overrides this slot with the continuous panel_size embedding
        # regardless of the in-sequence value.
        if 'gene_panel_value' not in item:
            item['gene_panel_value'] = torch.tensor([0])
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
            segment_start_idx = int((segment - 1) * self.seq_len_cell)
            segment_end_idx = int(segment * self.seq_len_cell)
            segment_tokens = item['gene_tokens'][
                segment_start_idx: segment_end_idx]
            if self.gt_type != 'rank':
                segment_values = item['gene_expr'][
                    segment_start_idx: segment_end_idx]
            else:
                segment_values = None
            if self.cell_pos_enc in _COORD_BASED_POS_ENCS:
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
                    torch.set_printoptions(threshold=float('inf'))
                    print(segment_tokens.size(0))
                    print(item['seg_tokens'])
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
                if self.cell_pos_enc in _COORD_BASED_POS_ENCS:
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

        # TODO: add special tokens from token dict directly (1 value per row)
        item['tissue_token'] = torch.tensor([103])
        item['assay_token'] = torch.tensor([104])
        item['gene_panel_token'] = torch.tensor([105])
        item['batch_token'] = torch.tensor([106])

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

        if self.cell_pos_enc in _COORD_BASED_POS_ENCS:
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
                if self.cell_pos_enc in _COORD_BASED_POS_ENCS:
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
            if self.cell_pos_enc in _COORD_BASED_POS_ENCS:
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
            if self.cell_pos_enc in _COORD_BASED_POS_ENCS:
                item_dict['rel_x_coords'] = F.pad(
                    item_dict['rel_x_coords'],
                    (0, pad_len),
                    value=float('-inf'))
                item_dict['rel_y_coords'] = F.pad(
                    item_dict['rel_y_coords'],
                    (0, pad_len),
                    value=float('-inf'))

        # Gene-panel subsampling + panel-size conditioning (no-op unless the
        # 'gene_panel' special token is configured). Must run on the assembled
        # gene-token sequence, before special tokens are prepended.
        item_dict = self._apply_gene_panel(item=item, item_dict=item_dict)

        # Add special tokens
        if self.n_special_tokens > 0:
            if self.pad_special_tokens:
                # IMPORTANT: each special token slot is ONE sequence
                # position. `_add_special_seq` iterates over
                # ``self.special_tokens`` and prepends
                # ``item[f'{spc_tk}_token']`` per iteration; the total
                # number of prepended tokens is therefore
                # ``len(special_tokens) * len(item[f'{spc_tk}_token'])``.
                # Per-slot vectors must be length 1 (not
                # n_special_tokens) so the prepend count matches the
                # n_special_tokens entries prepended to coords /
                # segments / positions. Using
                # `[0] * self.n_special_tokens` here happens to work
                # at n_special_tokens=1 (1*1=1) but produces an
                # n_special_tokens**2-sized prepend for >1, which
                # misaligns tokens vs coords at inference.
                for spc_tk in self.special_tokens:
                    # Exempt 'gene_panel': its token must stay non-zero so the
                    # slot is NOT masked out of attention (masks_attention is
                    # tokens != 0), and its panel-size signal must reach the
                    # model at inference. The panel size rides a separate
                    # non-maskable field (item_dict['panel_size'], set in
                    # _apply_gene_panel) consumed by the model's
                    # panel_size_embed, so the in-sequence value stays a
                    # placeholder.
                    if spc_tk == 'gene_panel':
                        continue
                    item[f'{spc_tk}_token'] = torch.tensor([0])
                    item[f'{spc_tk}_value'] = torch.tensor([0])
            item_dict = self._add_special_seq(item=item,
                                              item_dict=item_dict)

        # Add cell ID
        if self.include_cell_id:
            item_dict['cell_id'] = item['cell_id']

        if self.nz_spc:
            item_dict['segments'][
                (item_dict['segments'] != 0) & (
                    torch.arange(len(item_dict['segments'])
                    ) >= self.n_special_tokens)] += self.n_special_tokens
        #print(item_dict['tokens'])
        #print(item_dict['values'])
        #print(item_dict['segments'])
        #print(item_dict['positions'])

        # Expose per-cell metadata (batch_value, assay_value, ...) as
        # scalar fields independent of self.special_tokens.
        item_dict = self._attach_metadata(item=item, item_dict=item_dict)

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

    def _get_segment_seq(self,
                         item: int,
                         segment: int,
                         segment_seq_len: int,
                         ) -> tuple[list[int], list[float]]:
        """
        Get gene tokens and values for a given segment. Overrides the
        base class method to handle the two-segment layout (cell +
        aggregated neighborhood) used by the CellNeighborhoodTokenizer.

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
        segment_rel_x_coords:
            List of relative x coordinates for a given segment.
        segment_rel_y_coords:
            List of relative y coordinates for a given segment.
        """
        # Determine segment boundaries based on the two-segment layout
        if segment == 1:
            segment_start_idx = 0
            segment_end_idx = self.seq_len_cell
        elif segment == 2:
            segment_start_idx = self.seq_len_cell
            segment_end_idx = self.seq_len_cell + self.seq_len_neighborhood
        else:
            raise ValueError(
                f"CellNeighborhoodDataset only supports segments 1 and 2, "
                f"got {segment}.")

        segment_tokens = item['gene_tokens'][
            segment_start_idx:segment_end_idx]
        if self.gt_type != 'rank':
            segment_values = item['gene_expr'][
                segment_start_idx:segment_end_idx]
        else:
            segment_values = None
        if self.cell_pos_enc in _COORD_BASED_POS_ENCS:
            segment_rel_x_coords = item['rel_x_coord'][
                segment_start_idx:segment_end_idx]
            segment_rel_y_coords = item['rel_y_coord'][
                segment_start_idx:segment_end_idx]
        else:
            segment_rel_x_coords = None
            segment_rel_y_coords = None

        if segment != 1 and self.sep_gene_tokens_neb:
            segment_token_nz_mask = segment_tokens.ne(0)
            segment_tokens[segment_token_nz_mask] += self.vocab_size

        # Validate segment length
        if (self.sampling_strategy is not None and 'rep' in
        self.sampling_strategy):
            pass
        else:
            if segment_tokens.size(0) < segment_seq_len:
                raise ValueError(
                    'Sequence length for a given segment cannot be larger '
                    'than segment size when not sampling with replacement.')

        # If no sampling strategy is specified, use all tokens up to
        # specified length
        if self.sampling_strategy is None:
            segment_tokens = segment_tokens[:segment_seq_len]
            if self.gt_type != 'rank':
                segment_values = segment_values[:segment_seq_len]
            if self.cell_pos_enc in _COORD_BASED_POS_ENCS:
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

    def __getitem__(self,
                    item: int
                    ) -> dict:
        item_dict = {}

        # Retrieve Hugging Face item once
        item = self.dataset[item]

        # TODO: add special tokens from token dict directly (1 value per row)
        item['tissue_token'] = torch.tensor([103])
        item['assay_token'] = torch.tensor([104])
        item['gene_panel_token'] = torch.tensor([105])
        item['batch_token'] = torch.tensor([106])

        # Add segment to item (TODO: if statement to support old API)
        if 'seg_tokens' not in item.keys():
            seg_tokens = torch.cat([
                torch.ones(self.seq_len_cell, dtype=torch.long),
                torch.full((self.seq_len_neighborhood,), 2, dtype=torch.long),
            ])
            # Mask out positions where gene_tokens == 0
            seg_tokens = seg_tokens * (item['gene_tokens'] != 0).long()
            item['seg_tokens'] = seg_tokens
        
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
        if self.cell_pos_enc in _COORD_BASED_POS_ENCS:
            item_dict['rel_x_coords'] = torch.cat(
                [rel_x_coords_cell, rel_x_coords_neigh], dim=0)
            item_dict['rel_y_coords'] = torch.cat(
                [rel_y_coords_cell, rel_y_coords_neigh], dim=0)

        if self.gt_type != 'counts':
            item_dict['positions'] = torch.cat([
                torch.arange(1, gene_tokens_cell.size(0) + 1),
                torch.arange(1, gene_tokens_neigh.size(0) + 1)])
            item_dict['positions'] = item_dict['positions'] * (
                item_dict['tokens'] != 0).to(item_dict['positions'].dtype)

        if self.gt_type != 'rank':
            item_dict['values'] = torch.cat([values_cell, values_neigh], dim=0)

        # Gene-panel subsampling + panel-size conditioning (no-op unless the
        # 'gene_panel' special token is configured).
        item_dict = self._apply_gene_panel(item=item, item_dict=item_dict)

        # Add special tokens
        if self.n_special_tokens > 0:
            if self.pad_special_tokens:
                # See CellGraphDataset for the rationale: each
                # special-token slot is ONE sequence position, so the
                # per-slot vector must have length 1. Using
                # `[0] * n_special_tokens` here was the same off-by-N
                # bug -- works at n_special_tokens=1, breaks for >1.
                for spc_tk in self.special_tokens:
                    # Exempt 'gene_panel': its token must stay non-zero so the
                    # slot is NOT masked out of attention (masks_attention is
                    # tokens != 0), and its panel-size signal must reach the
                    # model at inference. The panel size rides a separate
                    # non-maskable field (item_dict['panel_size'], set in
                    # _apply_gene_panel) consumed by the model's
                    # panel_size_embed, so the in-sequence value stays a
                    # placeholder.
                    if spc_tk == 'gene_panel':
                        continue
                    item[f'{spc_tk}_token'] = torch.tensor([0])
                    item[f'{spc_tk}_value'] = torch.tensor([0])
            item_dict = self._add_special_seq(item=item,
                                              item_dict=item_dict)

        # Add cell ID
        if self.include_cell_id:
            item_dict['cell_id'] = item['cell_id']

        if self.nz_spc:
            item_dict['segments'][
                (item_dict['segments'] != 0) & (
                    torch.arange(len(item_dict['segments'])
                    ) >= self.n_special_tokens)] += self.n_special_tokens
        #print(item_dict['tokens'])
        #print(item_dict['values'])
        #print(item_dict['segments'])
        #print(item_dict['positions'])

        # Expose per-cell metadata (batch_value, assay_value, ...) as
        # scalar fields independent of self.special_tokens.
        item_dict = self._attach_metadata(item=item, item_dict=item_dict)

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