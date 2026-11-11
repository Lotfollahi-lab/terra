"""Mixed scopes against an independent tiny, scalar perturbation reference."""

import copy
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from datasets import Dataset

from terra import perturb_dataset

GENES = [f"ENSG{i:011d}" for i in range(1, 5)]
MODES = [(False, False), (True, False), (False, True), (True, True)]
TOKENS = [[1, 2, 1, 3, 2, 3], [1, 3, 2, 3, 1, 2], [2, 3, 1, 2, 1, 3]]
COUNTS = [
    [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    [30.0, 40.0, 50.0, 60.0, 10.0, 20.0],
    [50.0, 60.0, 10.0, 20.0, 30.0, 40.0],
]
IDS = [["A", "A", "B", "B", "C", "C"], ["B", "B", "C", "C", "A", "A"], ["C", "C", "A", "A", "B", "B"]]


def _row(cell="all", gene=1, target="cell", factor=None):
    return {
        "perturbed_cell_id": cell,
        "perturbed_ensembl_id": "all" if gene == "all" else GENES[gene - 1],
        "perturbation_target": target,
        "perturbation_type": "knockout" if factor is None else "foldchange",
        "foldchange": np.nan if factor is None else factor,
    }


@pytest.fixture
def inputs(tmp_path):
    (tmp_path / "token_dictionary.pkl").write_bytes(
        pickle.dumps({"<pad>": 0, **{gene: i + 1 for i, gene in enumerate(GENES)}})
    )
    return tmp_path


def _dataset(log=False, neighbors=True):
    data = {
        "cell_id": ["A", "B", "C"],
        "gene_tokens": TOKENS,
        "gene_expr": np.log1p(COUNTS).tolist() if log else COUNTS,
        "metadata": ["first", "second", "third"],
    }
    if neighbors:
        data["cell_ids"] = IDS
    dataset = Dataset.from_dict(data)
    dataset.set_format("torch", columns=["gene_tokens", "gene_expr"], output_all_columns=True)
    return dataset


def _expected(rows, log, pad, adjust):
    """Scalar reference: original-token eligibility, evolving-token edits."""
    tokens = copy.deepcopy(TOKENS)
    values = np.log1p(COUNTS).tolist() if log else copy.deepcopy(COUNTS)
    masks = [[False] * len(rows) for _ in TOKENS]
    for j, row in enumerate(rows):
        for b, cell in enumerate(["A", "B", "C"]):
            for p in range(6):
                own = row["perturbation_target"] == "cell"
                if (p < 2) != own:
                    continue
                if row["perturbed_cell_id"] not in ("all", cell if own else IDS[b][p]):
                    continue
                gene = row["perturbed_ensembl_id"]
                token = GENES.index(gene) + 1 if gene != "all" else None
                masks[b][j] |= gene == "all" or TOKENS[b][p] == token
                if gene != "all" and tokens[b][p] != token:
                    continue
                if row["perturbation_type"] == "knockout":
                    values[b][p] = 0.0
                    if pad:
                        tokens[b][p] = 0
                else:
                    v, factor = values[b][p], row["foldchange"]
                    values[b][p] = float(np.log1p(np.expm1(v) * factor)) if log else v * factor
    if adjust:
        for b in range(3):
            for start in range(0, 6, 2):
                order = sorted(range(start, start + 2), key=lambda p: tokens[b][p] == 0)
                tokens[b][start : start + 2] = [tokens[b][p] for p in order]
                values[b][start : start + 2] = [values[b][p] for p in order]
    return tokens, values, masks


def _check(bundle, rows, *, mode=(False, True), log=False, pad=True, adjust=False, neighbors=True):
    dataset = _dataset(log, neighbors)
    (bundle / "model_config.yaml").write_text(
        yaml.safe_dump({"data": {"count_count_norm_method": "shifted_log" if log else None}})
    )
    frame = pd.DataFrame(rows, index=[9] * len(rows))
    before, fmt, frame_before = dataset.to_dict(), copy.deepcopy(dataset.format), frame.copy(deep=True)
    tokens, values, masks = _expected(rows, log, pad, adjust)
    filtered, flags = mode
    selected = [i for i in range(3) if not filtered or any(masks[i])]
    result = perturb_dataset(
        dataset,
        frame,
        str(bundle),
        seq_len_cell=2,
        n_segments=3,
        nproc=None,
        batch_size=2,
        keep_in_memory=True,
        pad_gene_tokens=pad,
        adjust_positions=adjust,
        return_only_perturbed_cells=filtered,
        return_perturbation_flags=flags,
    )
    actual = result.with_format(None).to_dict()
    assert actual["gene_tokens"] == [tokens[i] for i in selected]
    np.testing.assert_allclose(actual["gene_expr"], [values[i] for i in selected], rtol=2e-6, atol=1e-7)
    for column in ["cell_id", "metadata"] + (["cell_ids"] if neighbors else []):
        assert actual[column] == [before[column][i] for i in selected]
    expected_columns = set(before)
    if flags:
        for j in range(len(rows)):
            column = f"pert_flag_idx_{j}"
            assert actual[column] == [masks[i][j] for i in selected]
            assert result[column][:].dtype == torch.bool
            expected_columns.add(column)
    assert set(actual) == expected_columns
    assert result.format["type"] == fmt["type"]
    assert result.format["output_all_columns"] == fmt["output_all_columns"]
    assert set(result.format["columns"]) == set(fmt["columns"]) | (expected_columns - set(before))
    assert dataset.to_dict() == before
    assert dataset.format == fmt
    pd.testing.assert_frame_equal(frame, frame_before)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("pad", [False, True])
def test_mixed_knockouts(inputs, mode, reverse, pad):
    rows = [_row(), _row("A", 2), _row("B", 3)]
    _check(inputs, rows[::-1] if reverse else rows, mode=mode, pad=pad)


@pytest.mark.parametrize("wildcard_target", ["cell", "neighborhood"])
@pytest.mark.parametrize("specific_target", ["cell", "neighborhood"])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_neighborhood_scopes(inputs, wildcard_target, specific_target, reverse):
    rows = [_row(target=wildcard_target), _row("B", 3, specific_target)]
    _check(inputs, rows[::-1] if reverse else rows, mode=(True, True))


@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_overlapping_repeated_fold_changes(inputs, log, reverse):
    rows = [_row(factor=0.5), _row("A", factor=2.0), _row(factor=4.0), _row("A", factor=2.0)]
    _check(inputs, rows[::-1] if reverse else rows, log=log)


@pytest.mark.parametrize("reverse", [False, True])
def test_row_order_with_real_fold_change_calls(inputs, monkeypatch, reverse):
    import terra.inference.perturb as module

    rows = [_row(factor=0.5), _row("A", factor=2.0), _row(factor=4.0), _row("A", factor=0.25)]
    if reverse:
        rows.reverse()
    calls = []
    original = module._fold_change

    def record(values, factor, log1p_values):
        calls.append(factor)
        return original(values, factor, log1p_values)

    monkeypatch.setattr(module, "_fold_change", record)
    _check(inputs, rows)
    # The first sequence is A: all four rows apply, in original order.
    assert calls[:4] == [row["foldchange"] for row in rows]


@pytest.mark.parametrize("mode", MODES)
def test_wildcard_neighborhood_needs_no_provenance(inputs, mode):
    _check(inputs, [_row(target="neighborhood"), _row("A", 2)], mode=mode, neighbors=False)


@pytest.mark.parametrize("mode", MODES)
def test_specific_neighborhood_still_requires_provenance(inputs, mode):
    with pytest.raises(ValueError, match="add_neigh_cell_ids=True"):
        _check(inputs, [_row(), _row("B", target="neighborhood")], mode=mode, neighbors=False)


@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("pad", [False, True])
def test_all_genes_and_pre_edit_eligibility(inputs, log, pad):
    # Later requests remain eligible even after a padded knockout removed a gene.
    _check(
        inputs,
        [_row("A", "all"), _row(factor=2.0), _row("B", 4), _row(gene="all", target="neighborhood", factor=0.5)],
        log=log,
        pad=pad,
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("neighbors", [False, True])
def test_mixed_adjust_positions_preserves_neighbor_segments(inputs, reverse, neighbors):
    rows = [_row(), _row("A", 4), _row(target="neighborhood")]
    if neighbors:
        rows.append(_row("B", 3, "neighborhood"))
    _check(inputs, rows[::-1] if reverse else rows, mode=(True, True), adjust=True, neighbors=neighbors)


@pytest.mark.parametrize("cell", ["all", "A"])
@pytest.mark.parametrize("target", ["cell", "neighborhood"])
def test_standalone_scopes_remain_unchanged(inputs, cell, target):
    _check(inputs, [_row(cell, target=target)])
