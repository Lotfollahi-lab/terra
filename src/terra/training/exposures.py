"""
Exposure logging, exposure counters and per-exposure loss weights for data
selection experiments (effective-exposure allocation on overlapping views).

Every cell of a cell-graph dataset enters a training sequence many times per
epoch: once as the centre of its own niche and once as a neighbour in the
niche of each cell that has it among its nearest neighbours. Each such
appearance is an *exposure*; it is a *target exposure* when at least one of
the cell's gene tokens is a prediction target. This module records, during
training and without changing it:

* per-niche losses: the mean target loss of every sequence trained on, with
  its row and global step;
* per-cell losses: the mean target loss of each cell segment of a sequence,
  for a tracked subset of cells (every `cell_loss_track_every`-th row, or the
  rows of a file);
* exposure counters per epoch and row: niches centred on the cell, niches the
  cell appears in, niches with a target token of the cell, and its target
  tokens;

and it can weight each target token's loss by a per-row weight, of the niche
(`loss_weight_mode: niche`) or of the target token's cell (`cell`).

Rows are rows of the training split (the dataset passed to `train`). Neighbour
rows come from the `neighbor_offsets` column that
`CellGraphTokenizer(add_spatial_index=True)` writes; cell-level records,
counters and cell-mode weights need it, and `check_neighbor_rows` checks
before training that the split keeps each tokenized file's rows together and
in order, which the offsets assume.

Config (all keys optional; the section is off when absent)::

    exposure:
      log_niche_losses: false
      log_cell_losses: false
      cell_loss_track_every: 100     # rows with row % every == 0 ...
      cell_loss_track_rows: null     # ... or the rows in this .npy file
      counters: false
      flush_every: 500               # steps between writes of the records
      loss_weights: null             # .npy of one weight per row
      loss_weight_mode: 'niche'      # or 'cell'
      loss_weight_clip: [0.5, 2.0]
      neighbor_row_check: true       # check_neighbor_rows before training
      log_dir: null                  # default <run folder>/exposures

Files (in `log_dir`)::

    manifest.json                               settings, rows, launches
    rank<r>/niche_losses_ep<e>_<chunk>.npz      step, row, loss, n_targets
    rank<r>/cell_losses_ep<e>_<chunk>.npz       step, niche_row, cell_row,
                                                segment, loss, n_targets
    counters_ep<e>.npz                          centre, present, target,
                                                target_tokens (summed over
                                                ranks)

`segment` is 1 for the niche's centre cell and 2.. for its neighbours
(nearest first). The loss of a token is the mean over the embedding
dimensions of the training loss (L1 or smooth L1), and a niche's loss is the
mean over its valid targets, as in the training loss: without
`excl_spc_from_loss`, special-token targets count towards a niche's loss and
`n_targets` but belong to no cell, so its cells' `n_targets` add up to fewer.
Niches without a valid target are recorded with loss 0 and `n_targets` 0.
When a launch starts at epoch e (a new run, or a resume), the records and
counters of epochs >= e are deleted first, also those of ranks beyond the
current world size, so an epoch is never recorded twice; `manifest.json`
lists the launches (start epoch, world size, iterations per epoch).
`load_records` reads the records back.

Memory: the counters (4 x int32), the tracked-row mask and the weights are
per-row tensors on every training device: about 105 MB at 5M rows, 2 GB at
96M rows.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULTS = dict(
    log_niche_losses=False,
    log_cell_losses=False,
    cell_loss_track_every=100,
    cell_loss_track_rows=None,
    counters=False,
    flush_every=500,
    loss_weights=None,
    loss_weight_mode='niche',
    loss_weight_clip=[0.5, 2.0],
    neighbor_row_check=True,
    log_dir=None,
)
COUNTER_NAMES = ('centre', 'present', 'target', 'target_tokens')
RECORD_KINDS = ('niche_losses', 'cell_losses')
_EPOCH_FILE = re.compile(r'_ep(\d+)(?:_\d+)?\.npz$')


def exposure_config(args: dict) -> dict | None:
    """
    The `exposure` section of a model config with defaults filled in, or
    `None` if it is absent or enables nothing.
    """
    section = args.get('exposure') or {}
    unknown = set(section) - set(DEFAULTS)
    if unknown:
        raise ValueError(
            f"Unknown keys in the exposure section: {sorted(unknown)}; known: "
            f"{sorted(DEFAULTS)}.")
    cfg = {**DEFAULTS, **section}
    if cfg['loss_weight_mode'] not in ('niche', 'cell'):
        raise ValueError(
            "exposure.loss_weight_mode must be 'niche' or 'cell', not "
            f"{cfg['loss_weight_mode']!r}.")
    clip = cfg['loss_weight_clip']
    if clip is not None and (len(clip) != 2 or not 0 < clip[0] <= clip[1]):
        raise ValueError(
            "exposure.loss_weight_clip must be [low, high] with "
            f"0 < low <= high, not {clip!r}.")
    if int(cfg['flush_every']) < 1:
        raise ValueError("exposure.flush_every must be at least 1.")
    if int(cfg['cell_loss_track_every']) < 1:
        raise ValueError("exposure.cell_loss_track_every must be at least 1.")
    if not (cfg['log_niche_losses'] or cfg['log_cell_losses']
            or cfg['counters'] or cfg['loss_weights']):
        return None
    return cfg


def needs_neighbor_rows(cfg: dict | None) -> bool:
    """Whether the settings attribute exposures to neighbour cells."""
    return cfg is not None and bool(
        cfg['log_cell_losses'] or cfg['counters']
        or (cfg['loss_weights'] and cfg['loss_weight_mode'] == 'cell'))


def _arrow_chunks(view, chunk_rows: int):
    """The rows of a dataset view as pyarrow tables of `chunk_rows` rows."""
    for start in range(0, len(view), chunk_rows):
        yield start, view[start:start + chunk_rows]


def check_neighbor_rows(dataset, n_check: int = 1000, seed: int = 0,
                        atol: float = 1e-2,
                        chunk_rows: int = 1_000_000) -> int:
    """
    Check that every row's neighbours are at its row plus its
    `neighbor_offsets`, as the cell datasets assume, and raise a ValueError
    otherwise.

    The offsets hold while the rows of each tokenized file stay together and
    in order (as after selecting whole files). Checked exactly on all rows:
    `cell_index` must run 0, 1, ... within each file (a run starts at 0, every
    other row is its predecessor's plus one) and every `cell_index + offset`
    must lie in the row's run. Checked on `n_check` random rows, where the
    dataset has the coordinates: the neighbour rows' positions are the row's
    plus its stored relative coordinates.

    Returns the number of neighbour pairs checked exactly.
    """
    import pyarrow.compute as pc
    n_rows = len(dataset)
    if n_rows == 0:
        return 0
    if 'neighbor_offsets' not in dataset.column_names:
        raise ValueError(
            "The dataset has no neighbor_offsets column: tokenize it with "
            "CellGraphTokenizer(add_spatial_index=True).")
    problems = []

    # Exact: runs of cell_index and offsets inside them
    view = dataset.select_columns(['cell_index', 'neighbor_offsets']) \
        .with_format('arrow')
    cell_index = np.empty(n_rows, dtype=np.int64)
    for start, table in _arrow_chunks(view, chunk_rows):
        cell_index[start:start + table.num_rows] = \
            table.column('cell_index').to_numpy()
    continues = np.zeros(n_rows, dtype=bool)
    continues[1:] = cell_index[1:] == cell_index[:-1] + 1
    broken = np.flatnonzero(~((cell_index == 0) | continues))
    if broken.size:
        problems.append(
            f'{broken.size} rows break the runs of cell_index (first at row '
            f'{broken[0]}: cell_index {cell_index[broken[0]]} after '
            f'{cell_index[broken[0] - 1] if broken[0] else "the start"})')
    run_starts = np.flatnonzero(cell_index == 0)
    if run_starts.size == 0:
        raise ValueError(
            "Neighbour rows are not at row + neighbor_offsets: no row has "
            "cell_index 0, so the training split holds no file's first row.")
    run_id = np.cumsum(cell_index == 0) - 1
    run_length = np.diff(np.append(run_starts, n_rows))
    n_pairs, n_outside = 0, 0
    for start, table in _arrow_chunks(view, chunk_rows):
        offsets = table.column('neighbor_offsets').combine_chunks()
        values = offsets.flatten().to_numpy().astype(np.int64)
        parents = pc.list_parent_indices(offsets).to_numpy() + start
        target = cell_index[parents] + values
        length = run_length[np.maximum(run_id[parents], 0)]
        outside = (target < 0) | (target >= length) | (run_id[parents] < 0)
        n_outside += int(outside.sum())
        n_pairs += values.size
    if n_outside:
        problems.append(f'{n_outside} of {n_pairs} neighbour offsets point '
                        'outside their file\'s rows')

    # Sampled: coordinates
    coords = ('x_coord', 'y_coord', 'rel_x_coord', 'rel_y_coord')
    if not problems and n_check > 0 and all(
            c in dataset.column_names for c in coords):
        sample_view = dataset.select_columns(
            ['neighbor_offsets', *coords]).with_format(None)
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(n_rows, size=min(n_check, n_rows),
                                  replace=False))
        xy = sample_view.select_columns(['x_coord', 'y_coord'])
        for row, item in zip(rows, sample_view.select(rows)):
            offsets = np.asarray(item['neighbor_offsets'], dtype=np.int64)
            rel_x = np.asarray(item['rel_x_coord'], dtype=np.float64)
            rel_y = np.asarray(item['rel_y_coord'], dtype=np.float64)
            if offsets.size == 0 or rel_x.size != offsets.size + 1:
                continue
            neighbors = xy.select((row + offsets).tolist())
            for axis, rel in (('x_coord', rel_x), ('y_coord', rel_y)):
                delta = (np.asarray(neighbors[axis], dtype=np.float64)
                         - float(item[axis]))
                if (np.abs(delta - rel[1:])
                        > atol * np.maximum(1.0, np.abs(rel[1:]))).any():
                    problems.append(f'row {row}: neighbour positions do not '
                                    'match the relative coordinates')
                    break
            if len(problems) >= 5:
                break
    if problems:
        raise ValueError(
            "Neighbour rows are not at row + neighbor_offsets: the training "
            "split does not keep the rows of each tokenized file together "
            "and in order, so cell-level exposure records would be "
            "attributed to the wrong cells.\n  " + "\n  ".join(problems))
    return n_pairs


def load_weights(file_path: str, n_rows: int,
                 clip: list[float] | None) -> np.ndarray:
    """Per-row loss weights from a `.npy` file, clipped to `clip`."""
    weights = np.load(file_path).astype(np.float32).reshape(-1)
    if weights.size != n_rows:
        raise ValueError(
            f"exposure.loss_weights '{file_path}' holds {weights.size} weights,"
            f" but the training split has {n_rows} rows.")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError(
            f"exposure.loss_weights '{file_path}' has negative or non-finite "
            "weights.")
    if clip is not None:
        weights = np.clip(weights, clip[0], clip[1])
    return weights


def _token_loss(zi: torch.Tensor, hi: torch.Tensor,
                loss_fn_type: str) -> torch.Tensor:
    """Elementwise training loss [B, L, D], as the training loop computes it."""
    if loss_fn_type == 'smooth_l1':
        return F.smooth_l1_loss(zi.float(), hi.float(), reduction='none')
    if loss_fn_type == 'l1':
        return torch.abs(zi.float() - hi.float())
    raise ValueError(f'Unknown loss_fn_type {loss_fn_type}.')


def _record_epoch(path: str) -> int | None:
    match = _EPOCH_FILE.search(os.path.basename(path))
    return int(match.group(1)) if match else None


class ExposureTracker:
    """
    Records exposures and losses during training and computes weighted
    losses (see the module docstring). One per rank; all ranks call every
    method in the same order (`end_epoch` sums the counters over the ranks).
    Recording does not synchronise the device with the host; the buffered
    records are copied to the host every `flush_every` steps.

    Parameters
    ----------
    cfg:
        Settings (`exposure_config`).
    n_rows:
        Rows of the training split.
    n_segments:
        Segments of a sequence (the cell and its neighbours) the model reads.
    n_special_tokens, nz_spc:
        Special tokens before the gene tokens, and whether the cells'
        segment ids are shifted by their number (`data.nz_spc`).
    excl_spc_from_loss, loss_fn_type:
        As in the training loop.
    has_neighbor_rows:
        Whether the batches hold `neighbor_rows`.
    folder, rank, world_size, device:
        Run folder, distributed rank and size, and training device.
    manifest:
        Extra entries for `manifest.json` (rank 0 writes it).
    """
    def __init__(self, cfg: dict, n_rows: int, n_segments: int,
                 n_special_tokens: int, nz_spc: bool,
                 excl_spc_from_loss: bool, loss_fn_type: str,
                 has_neighbor_rows: bool, folder: str | None, rank: int,
                 world_size: int, device: torch.device,
                 manifest: dict | None = None):
        if needs_neighbor_rows(cfg) and not has_neighbor_rows:
            raise ValueError(
                "exposure.log_cell_losses, exposure.counters and "
                "loss_weight_mode 'cell' need the rows of each niche's "
                "neighbours: tokenize the dataset with "
                "CellGraphTokenizer(add_spatial_index=True) (cell-graph "
                "datasets only).")
        self.cfg = cfg
        self.n_rows = int(n_rows)
        # Cells with rows per niche (the batches' rows set it per batch)
        self.n_cells = int(n_segments) if has_neighbor_rows else 1
        self.n_special_tokens = int(n_special_tokens)
        self.seg_offset = int(n_special_tokens) if nz_spc else 0
        self.excl_spc_from_loss = excl_spc_from_loss
        self.loss_fn_type = loss_fn_type
        self.has_neighbor_rows = has_neighbor_rows
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.flush_every = int(cfg['flush_every'])
        self.logging = bool(cfg['log_niche_losses'] or cfg['log_cell_losses']
                            or cfg['counters'])
        self.log_dir = cfg['log_dir'] or (
            os.path.join(folder, 'exposures') if folder else None)
        if self.logging and self.log_dir is None:
            raise ValueError(
                "Exposure logging needs a run folder or exposure.log_dir.")
        self.rank_dir = (os.path.join(self.log_dir, f'rank{rank}')
                         if self.logging else None)
        if self.rank_dir:
            os.makedirs(self.rank_dir, exist_ok=True)

        # Tracked cells for the per-cell loss records
        self.tracked = None
        if cfg['log_cell_losses']:
            tracked = torch.zeros(self.n_rows, dtype=torch.bool)
            if cfg['cell_loss_track_rows']:
                rows = np.load(cfg['cell_loss_track_rows']).astype(
                    np.int64).reshape(-1)
                if rows.size and (rows.min() < 0 or rows.max() >= self.n_rows):
                    raise ValueError(
                        "exposure.cell_loss_track_rows holds rows outside "
                        f"0..{self.n_rows - 1}.")
                tracked[torch.from_numpy(rows)] = True
            else:
                tracked[::int(cfg['cell_loss_track_every'])] = True
            self.tracked = tracked.to(device)

        # Loss weights
        self.weights = None
        self.weight_stats = None
        if cfg['loss_weights']:
            weights = load_weights(cfg['loss_weights'], self.n_rows,
                                   cfg['loss_weight_clip'])
            self.weight_stats = dict(mean=float(weights.mean()),
                                     min=float(weights.min()),
                                     max=float(weights.max()))
            self.weights = torch.from_numpy(weights).to(device)

        self.counters = None
        if cfg['counters']:
            self.counters = {
                name: torch.zeros(self.n_rows, dtype=torch.int32,
                                  device=device)
                for name in COUNTER_NAMES}

        self.epoch = None
        self._launch_epoch = None
        self._buffers = {kind: [] for kind in RECORD_KINDS}
        self._chunks = {kind: 0 for kind in RECORD_KINDS}
        self._steps_since_flush = 0

        self._manifest = None
        if rank == 0 and self.logging:
            os.makedirs(self.log_dir, exist_ok=True)
            launches = []
            manifest_path = os.path.join(self.log_dir, 'manifest.json')
            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    launches = json.load(f).get('launches', [])
            self._manifest = dict(
                n_rows=self.n_rows,
                n_segments_with_rows=self.n_cells,
                segment_offset=self.seg_offset,
                n_special_tokens=self.n_special_tokens,
                excl_spc_from_loss=bool(excl_spc_from_loss),
                loss_fn_type=loss_fn_type,
                world_size=world_size,
                n_tracked_cells=(None if self.tracked is None
                                 else int(self.tracked.sum())),
                loss_weight_stats=self.weight_stats,
                config=cfg,
                **(manifest or {}),
                launches=launches)
            self._write_manifest()

    def _write_manifest(self):
        with open(os.path.join(self.log_dir, 'manifest.json'), 'w') as f:
            json.dump(self._manifest, f, indent=2, default=str)

    @property
    def weighted(self) -> bool:
        return self.weights is not None

    # -- per-batch helpers -------------------------------------------------
    def _cell_rows(self, rows: torch.Tensor,
                   neighbor_rows: torch.Tensor | None) -> torch.Tensor:
        """Rows of the cells of each niche [B, n_cells] (-1: no cell)."""
        if self.has_neighbor_rows and neighbor_rows is not None:
            return torch.cat([rows[:, None], neighbor_rows], dim=1)
        return rows[:, None]

    def _local_segments(self, segments: torch.Tensor,
                        n_cells: int) -> torch.Tensor:
        """
        Cell slot of each position: 1 = centre, 2.. = neighbours, 0 = no
        cell with a row (padding, special tokens, neighbours beyond the
        `n_cells` of the batch's cell rows).
        """
        local = segments.long() - self.seg_offset
        return torch.where((local >= 1) & (local <= n_cells), local,
                           torch.zeros_like(local))

    def _targets(self, masks_pred: list[torch.Tensor],
                 masks_pred_valid: torch.Tensor | None,
                 pad_special_tokens: bool):
        """
        Per target mask: the index slice the loss uses and its validity, both
        [B, L] (special tokens left out as in the loss).
        """
        out = []
        for m, idx in enumerate(masks_pred):
            valid = (torch.ones_like(idx, dtype=torch.bool)
                     if masks_pred_valid is None else masks_pred_valid[m])
            start = (self.n_special_tokens
                     if self.excl_spc_from_loss and not pad_special_tokens
                     else 0)
            out.append((start, idx[:, start:], valid[:, start:]))
        return out

    # -- weighted loss -----------------------------------------------------
    def weighted_loss(self, z: list[torch.Tensor], h: list[torch.Tensor],
                      masks_pred: list[torch.Tensor],
                      masks_pred_valid: torch.Tensor | None,
                      segments: torch.Tensor, rows: torch.Tensor,
                      neighbor_rows: torch.Tensor | None,
                      pad_special_tokens: bool) -> torch.Tensor:
        """
        The training loss with every target token's loss weighted by its
        row's weight: per target mask, the weighted mean over the valid
        targets (and embedding dimensions), ``sum(w * loss) / sum(w)``,
        summed over the target masks (the training loop divides by their
        number, as for the unweighted loss). Only the weights' ratios within a
        batch matter: constant weights give the unweighted loss. Tokens
        without a cell (special tokens in cell mode) get weight 1.
        """
        cell_rows = self._cell_rows(rows, neighbor_rows)
        loss = 0.
        for zi, hi, (start, idx, valid) in zip(
                z, h, self._targets(masks_pred, masks_pred_valid,
                                    pad_special_tokens)):
            zi, hi = zi[:, start:, :], hi[:, start:, :]
            token_loss = _token_loss(zi, hi, self.loss_fn_type)
            if self.cfg['loss_weight_mode'] == 'niche':
                w = self.weights[rows][:, None].expand_as(idx)
            else:
                slot = self._local_segments(segments.gather(1, idx),
                                            cell_rows.size(1))
                cell = cell_rows.gather(1, (slot - 1).clamp_min(0))
                known = (slot > 0) & (cell >= 0)
                w = torch.where(known, self.weights[cell.clamp_min(0)],
                                torch.ones_like(token_loss[..., 0]))
            weight = valid.to(token_loss.dtype) * w
            loss = loss + (token_loss * weight.unsqueeze(-1)).sum() / (
                weight.sum() * token_loss.size(-1)).clamp_min(
                    torch.finfo(token_loss.dtype).tiny)
        return loss

    # -- recording ---------------------------------------------------------
    def start_epoch(self, epoch: int):
        """
        Reset the counters and remove the records of `epoch`; at the first
        epoch of a launch, also those of all later epochs and, on rank 0, of
        ranks beyond the current world size and the counters.
        """
        self.epoch = int(epoch)
        self._chunks = {k: 0 for k in self._chunks}
        self._buffers = {k: [] for k in self._buffers}
        self._steps_since_flush = 0
        if self.counters is not None:
            for counter in self.counters.values():
                counter.zero_()
        if not self.logging:
            return
        first = self._launch_epoch is None
        if first:
            self._launch_epoch = self.epoch
        self._remove_records(self.rank_dir, first)
        if first and self.rank == 0:
            for rank_dir in glob.glob(os.path.join(self.log_dir, 'rank*')):
                rank = rank_dir[len(os.path.join(self.log_dir, 'rank')):]
                if rank.isdigit() and int(rank) >= self.world_size:
                    self._remove_records(rank_dir, True)
            for path in glob.glob(os.path.join(self.log_dir,
                                               'counters_ep*.npz')):
                if (_record_epoch(path) or 0) >= self.epoch:
                    os.remove(path)
            self._manifest['launches'].append(dict(
                start_epoch=self.epoch, world_size=self.world_size,
                iterations_per_epoch=self._manifest.get(
                    'iterations_per_epoch'),
                time=time.strftime('%Y-%m-%d %H:%M:%S')))
            self._write_manifest()

    def _remove_records(self, rank_dir: str, later_epochs: bool):
        for kind in RECORD_KINDS:
            for path in glob.glob(os.path.join(rank_dir, f'{kind}_ep*.npz')):
                epoch = _record_epoch(path)
                if epoch is not None and (
                        epoch == self.epoch
                        or (later_epochs and epoch > self.epoch)):
                    os.remove(path)

    @torch.no_grad()
    def record(self, step: int, z: list[torch.Tensor], h: list[torch.Tensor],
               masks_pred: list[torch.Tensor],
               masks_pred_valid: torch.Tensor | None,
               segments: torch.Tensor, rows: torch.Tensor,
               neighbor_rows: torch.Tensor | None,
               pad_special_tokens: bool):
        """Record the losses and exposures of one batch."""
        if not self.logging:
            return
        cfg = self.cfg
        B = rows.size(0)
        device = rows.device
        cell_rows = self._cell_rows(rows, neighbor_rows)
        n_cells = cell_rows.size(1)
        niche_sum = torch.zeros(B, device=device, dtype=torch.float32)
        niche_cnt = torch.zeros(B, device=device, dtype=torch.float32)
        cell_sum = torch.zeros(B, n_cells + 1, device=device,
                               dtype=torch.float32)
        cell_cnt = torch.zeros_like(cell_sum)
        for zi, hi, (start, idx, valid) in zip(
                z, h, self._targets(masks_pred, masks_pred_valid,
                                    pad_special_tokens)):
            tl = _token_loss(zi[:, start:, :], hi[:, start:, :],
                             self.loss_fn_type).mean(-1)  # [B, L]
            v = valid.to(torch.float32)
            niche_sum += (tl * v).sum(1)
            niche_cnt += v.sum(1)
            slot = self._local_segments(segments.gather(1, idx), n_cells)
            cell_sum.scatter_add_(1, slot, tl * v)
            cell_cnt.scatter_add_(1, slot, v)
        cell_sum, cell_cnt = cell_sum[:, 1:], cell_cnt[:, 1:]
        known = cell_rows >= 0
        step_rows = torch.full((B,), int(step), dtype=torch.int64,
                               device=device)

        if cfg['log_niche_losses']:
            self._buffers['niche_losses'].append(dict(
                step=step_rows, row=rows.long(),
                loss=niche_sum / niche_cnt.clamp_min(1),
                n_targets=niche_cnt.to(torch.int32)))

        if cfg['log_cell_losses']:
            # Fixed shapes [B, n_cells]; the records are selected on the host
            self._buffers['cell_losses'].append(dict(
                step=step_rows, niche_row=rows.long(), cell_row=cell_rows,
                loss=cell_sum / cell_cnt.clamp_min(1),
                n_targets=cell_cnt.to(torch.int32),
                select=(cell_cnt > 0) & known
                & self.tracked[cell_rows.clamp_min(0)]))

        if self.counters is not None:
            # Masked additions at row 0 for unknown cells add nothing
            flat_rows = cell_rows.clamp_min(0).flatten()
            self.counters['centre'].index_add_(
                0, rows.long(), torch.ones(B, dtype=torch.int32,
                                           device=device))
            slot_all = self._local_segments(segments, n_cells)
            present = torch.zeros(B, n_cells + 1, dtype=torch.bool,
                                  device=device)
            present.scatter_(1, slot_all, True)
            present = present[:, 1:] & known
            hit = (cell_cnt > 0) & known
            self.counters['present'].index_add_(
                0, flat_rows, present.flatten().to(torch.int32))
            self.counters['target'].index_add_(
                0, flat_rows, hit.flatten().to(torch.int32))
            self.counters['target_tokens'].index_add_(
                0, flat_rows, (cell_cnt * hit).flatten().to(torch.int32))

        self._steps_since_flush += 1
        if self._steps_since_flush >= self.flush_every:
            self.flush()

    def flush(self):
        """Write the buffered records of this rank."""
        self._steps_since_flush = 0
        for kind, buffer in self._buffers.items():
            if not buffer:
                continue
            arrays = {key: torch.cat([b[key] for b in buffer]).cpu().numpy()
                      for key in buffer[0]}
            buffer.clear()
            if kind == 'cell_losses':
                b, s = np.nonzero(arrays.pop('select'))
                arrays = dict(step=arrays['step'][b],
                              niche_row=arrays['niche_row'][b],
                              cell_row=arrays['cell_row'][b, s],
                              segment=(s + 1).astype(np.int16),
                              loss=arrays['loss'][b, s],
                              n_targets=arrays['n_targets'][b, s])
            path = os.path.join(
                self.rank_dir,
                f'{kind}_ep{self.epoch:03d}_{self._chunks[kind]:05d}.npz')
            np.savez(path, **arrays)
            self._chunks[kind] += 1

    def end_epoch(self, epoch: int):
        """Write the epoch's records and its counters summed over ranks."""
        if not self.logging:
            return
        self.flush()
        if self.counters is None:
            return
        for counter in self.counters.values():
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(counter, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            np.savez_compressed(
                os.path.join(self.log_dir, f'counters_ep{int(epoch):03d}.npz'),
                **{name: counter.cpu().numpy()
                   for name, counter in self.counters.items()})


def load_records(log_dir: str, kind: str = 'niche_losses',
                 epoch: int | None = None) -> dict[str, np.ndarray]:
    """
    The records of all ranks (`kind` 'niche_losses' or 'cell_losses'), one
    epoch or all, concatenated in rank and chunk order.
    """
    pattern = (f'{kind}_ep{epoch:03d}_*.npz' if epoch is not None
               else f'{kind}_ep*_*.npz')

    def rank_then_name(path):
        rank_dir = os.path.basename(os.path.dirname(path))
        return int(rank_dir[len('rank'):]), os.path.basename(path)
    paths = sorted(glob.glob(os.path.join(log_dir, 'rank*', pattern)),
                   key=rank_then_name)
    if not paths:
        return {}
    parts = [dict(np.load(p)) for p in paths]
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
