"""Perturbation gene validation against a tiny model vocabulary, without weights."""

import pickle

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from datasets import Dataset

from terra import perturb_dataset

GENE = "ENSG00000000001"
ABSENT_GENE = "ENSG00000000003"
RETURN_MODES = [(False, False), (True, False), (False, True), (True, True)]


@pytest.fixture
def perturb_inputs(tmp_path):
    with (tmp_path / "token_dictionary.pkl").open("wb") as handle:
        pickle.dump({"<pad>": 0, GENE: 1, "ENSG00000000002": 2, ABSENT_GENE: 3}, handle)
    (tmp_path / "model_config.yaml").write_text(yaml.safe_dump({"data": {"count_count_norm_method": None}}))
    dataset = Dataset.from_dict(
        {
            "cell_id": ["A", "B"],
            "gene_tokens": [[1, 2, 1, 2], [1, 0, 2, 1]],
            "gene_expr": [[1.0, 2.0, 3.0, 4.0], [5.0, 0.0, 7.0, 8.0]],
        }
    )
    dataset.set_format("torch", columns=["gene_tokens", "gene_expr"], output_all_columns=True)
    return dataset, tmp_path


def _frame(genes, cells="all"):
    return pd.DataFrame(
        {
            "perturbed_cell_id": cells,
            "perturbed_ensembl_id": genes,
            "perturbation_target": "cell",
            "perturbation_type": "knockout",
            "foldchange": np.nan,
        }
    )


def _run(inputs, frame, filtered=False, flags=False):
    dataset, bundle = inputs
    return perturb_dataset(
        dataset,
        frame,
        model_folder_path=str(bundle),
        seq_len_cell=2,
        n_segments=2,
        nproc=None,  # Run the Dataset.map in-process, without multiprocessing.
        return_only_perturbed_cells=filtered,
        return_perturbation_flags=flags,
    )


@pytest.mark.parametrize("filtered, flags", RETURN_MODES)
@pytest.mark.parametrize("cells", ["all", "A"])
def test_unknown_gene_raises_for_every_return_mode(perturb_inputs, filtered, flags, cells):
    frame = _frame(["ENSG_UNKNOWN"], cells)
    with pytest.raises(ValueError, match="ENSG_UNKNOWN"):
        _run(perturb_inputs, frame, filtered, flags)


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA], ids=["None", "NaN", "NA"])
def test_missing_gene_id_is_rejected(perturb_inputs, missing):
    with pytest.raises(ValueError, match="missing"):
        _run(perturb_inputs, _frame([missing]))


def test_mixed_invalid_rows_raise_before_mapping_without_mutating_inputs(perturb_inputs, monkeypatch):
    dataset, _ = perturb_inputs
    frame = _frame([GENE, "ENSG_UNKNOWN", "all", "ENSG_OTHER_UNKNOWN", None])
    frame.index = [9, 9, -1, 2, 2]
    before = frame.copy(deep=True)
    original_data, original_format = dataset.to_dict(), dataset.format

    def unexpected_map(*args, **kwargs):
        pytest.fail("Invalid perturbations must be rejected before Dataset.map")

    monkeypatch.setattr(Dataset, "map", unexpected_map)
    with pytest.raises(ValueError) as exc:
        _run(perturb_inputs, frame)
    message = str(exc.value)
    for text in ("ENSG_UNKNOWN", "ENSG_OTHER_UNKNOWN", "None", "vocabulary", "Ensembl", "all"):
        assert text in message
    pd.testing.assert_frame_equal(frame, before)
    assert dataset.to_dict() == original_data
    assert dataset.format == original_format


@pytest.mark.parametrize("filtered, flags", RETURN_MODES)
@pytest.mark.parametrize("gene", [GENE, "all"])
@pytest.mark.parametrize("cells", ["all", "A"])
def test_valid_and_wildcard_knockouts_are_unchanged(perturb_inputs, filtered, flags, gene, cells):
    dataset, _ = perturb_inputs
    frame = _frame([gene], cells)
    before = frame.copy(deep=True)
    original_data, original_format = dataset.to_dict(), dataset.format
    expected_tokens = dataset["gene_tokens"][:].clone()
    expected_values = dataset["gene_expr"][:].clone()
    edited = torch.zeros_like(expected_tokens, dtype=torch.bool)
    edited[:, :2] = True if gene == "all" else expected_tokens[:, :2] == 1
    if cells == "A":
        edited[1] = False
    expected_tokens[edited], expected_values[edited] = 0, 0
    affected = edited.any(dim=1)

    result = _run(perturb_inputs, frame, filtered, flags)
    selected = affected if filtered else torch.ones(2, dtype=torch.bool)
    torch.testing.assert_close(result["gene_tokens"][:], expected_tokens[selected], rtol=0, atol=0)
    torch.testing.assert_close(result["gene_expr"][:], expected_values[selected], rtol=0, atol=0)
    assert result["cell_id"][:] == [cell for cell, keep in zip(["A", "B"], selected, strict=True) if keep]
    if flags:
        torch.testing.assert_close(result["pert_flag_idx_0"][:], affected[selected])
    pd.testing.assert_frame_equal(frame, before)
    assert dataset.to_dict() == original_data
    assert dataset.format == original_format


@pytest.mark.parametrize("filtered, flags", RETURN_MODES)
@pytest.mark.parametrize("gene, token", [(ABSENT_GENE, 3), ("ENSG00000000002", 2)])
def test_vocabulary_gene_absent_from_cells_is_allowed(perturb_inputs, filtered, flags, gene, token):
    dataset, _ = perturb_inputs
    affected = (dataset["gene_tokens"][:][:, :2] == token).any(dim=1)
    result = _run(perturb_inputs, _frame([gene]), filtered, flags)
    assert len(result) == (int(affected.sum()) if filtered else len(dataset))
    if not affected.any() and not filtered:
        torch.testing.assert_close(result["gene_tokens"][:], dataset["gene_tokens"][:], rtol=0, atol=0)
        torch.testing.assert_close(result["gene_expr"][:], dataset["gene_expr"][:], rtol=0, atol=0)
    if not filtered:
        # Cell B lacks both genes in its own segment, even though gene 2 is
        # present in its neighborhood. Its intrinsic expression stays intact.
        torch.testing.assert_close(result["gene_expr"][1], dataset["gene_expr"][1], rtol=0, atol=0)
    if flags and len(result):
        expected = affected[affected] if filtered else affected
        torch.testing.assert_close(result["pert_flag_idx_0"][:], expected)
