"""Linear probes of ``terra.evaluation``: early stopping restores the best
epoch, optional seeding, and the Jensen-Shannon metrics of the regressor."""
import json
import logging
import re

import numpy as np
import pytest
import scipy.spatial.distance as dist
import torch
import torch.nn as nn

from terra.evaluation.linear import (LinearClassifier, linear_classifier,
                                     linear_regressor)


def _data(seed=0, n_train=60, n_val=60, n_test=100, n_features=50,
          n_classes=4):
    """Few, noisy samples in many dimensions: the probe overfits within a few
    epochs at a high learning rate, so early stopping triggers."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, n_features)) * 0.2

    def draw(n):
        y = rng.integers(0, n_classes, n)
        x = centers[y] + rng.normal(size=(n, n_features))
        return x.astype(np.float32), y

    return draw(n_train), draw(n_val), draw(n_test)


def _val_losses(caplog):
    return [float(m) for m in re.findall(r'Val Loss = ([0-9.]+)', caplog.text)]


def _val_loss(model, x, y):
    model.eval()
    with torch.no_grad():
        return float(nn.CrossEntropyLoss()(
            model(torch.tensor(x)), torch.tensor(y)))


def test_state_dict_entries_are_live_references():
    # Why a snapshot is needed: state_dict() follows in-place updates
    model = LinearClassifier(4, 2)
    state = model.state_dict()
    before = state['linear.weight'].clone()
    with torch.no_grad():
        model.linear.weight.add_(1.)
    assert not torch.equal(state['linear.weight'], before)


@pytest.mark.parametrize('restore_best', [True, False])
def test_linear_classifier_restores_best_epoch(caplog, restore_best):
    (xtr, ytr), (xva, yva), (xte, yte) = _data()
    caplog.set_level(logging.INFO, logger='terra.evaluation.linear')
    # One validation batch, so the logged validation loss is the full-set loss
    _, _, _, model = linear_classifier(
        xtr, ytr, xva, yva, xte, yte, n_epochs=100, batch_size=64, lr=4.,
        patience=3, restore_best=restore_best, seed=0)
    losses = _val_losses(caplog)
    assert 'Early stopping' in caplog.text
    assert min(losses) < losses[-1] - 0.05
    returned = _val_loss(model, xva, yva)
    if restore_best:
        # The weights of the epoch with the lowest validation loss
        assert returned == pytest.approx(min(losses), abs=1e-4)
    else:
        # Earlier behaviour: the weights of the last epoch
        assert returned == pytest.approx(losses[-1], abs=1e-4)


def test_linear_classifier_without_validation_set_uses_last_epoch(caplog):
    (xtr, ytr), _, (xte, yte) = _data()
    preds, targets, logits, model = linear_classifier(
        xtr, ytr, None, None, xte, yte, n_epochs=5, batch_size=64, seed=0)
    assert len(preds) == len(targets) == len(logits) == len(yte)


def test_linear_classifier_seed():
    (xtr, ytr), (xva, yva), (xte, yte) = _data()
    kwargs = dict(n_epochs=20, batch_size=16, lr=0.5, patience=50)
    torch.manual_seed(1)
    a = linear_classifier(xtr, ytr, xva, yva, xte, yte, seed=3, **kwargs)
    torch.manual_seed(2)
    b = linear_classifier(xtr, ytr, xva, yva, xte, yte, seed=3, **kwargs)
    # Same seed: identical results, whatever the global random state
    np.testing.assert_array_equal(np.stack(a[2]), np.stack(b[2]))
    c = linear_classifier(xtr, ytr, xva, yva, xte, yte, seed=4, **kwargs)
    assert not np.array_equal(np.stack(a[2]), np.stack(c[2]))
    # No seed: the global random state decides, as before
    torch.manual_seed(5)
    d = linear_classifier(xtr, ytr, xva, yva, xte, yte, **kwargs)
    torch.manual_seed(5)
    e = linear_classifier(xtr, ytr, xva, yva, xte, yte, **kwargs)
    np.testing.assert_array_equal(np.stack(d[2]), np.stack(e[2]))


def _compositions(y, n_classes=4):
    comp = np.full((len(y), n_classes), 0.1 / n_classes, dtype=np.float32)
    comp[np.arange(len(y)), y] += 0.9
    return comp


@pytest.mark.parametrize('restore_best', [True, False])
def test_linear_regressor_restores_best_epoch(caplog, restore_best):
    (xtr, ytr), (xva, yva), (xte, yte) = _data()
    caplog.set_level(logging.INFO, logger='terra.evaluation.linear')
    _, model = linear_regressor(
        xtr, _compositions(ytr), xva, _compositions(yva), xte,
        _compositions(yte), n_epochs=200, batch_size=64, lr=4., patience=3,
        restore_best=restore_best, seed=0)
    losses = _val_losses(caplog)
    assert 'Early stopping' in caplog.text
    model.eval()
    with torch.no_grad():
        returned = float(nn.MSELoss()(model(torch.tensor(xva)),
                                      torch.tensor(_compositions(yva))))
    expected = min(losses) if restore_best else losses[-1]
    assert returned == pytest.approx(expected, abs=1e-4)


def test_linear_regressor_jensen_shannon_metrics(tmp_path, caplog):
    (xtr, ytr), (xva, yva), (xte, yte) = _data()
    targets = _compositions(yte)
    path = tmp_path / 'metrics.json'
    preds, _ = linear_regressor(
        xtr, _compositions(ytr), xva, _compositions(yva), xte, targets,
        n_epochs=5, batch_size=64, seed=0, results_save_path=str(path))
    metrics = json.loads(path.read_text())
    js = dist.jensenshannon(targets, preds, axis=-1)
    # 'JSD' is the Jensen-Shannon distance (as before), now also stored under
    # an explicit name, next to the divergence
    assert metrics['JSD'] == pytest.approx(float(js.mean()))
    assert metrics['JS_distance'] == metrics['JSD']
    assert metrics['JS_divergence'] == pytest.approx(float((js ** 2).mean()),
                                                     rel=1e-5)

    # An all-zero composition (undefined distance) is left out of the
    # averages instead of making them NaN
    targets[0] = 0.
    caplog.set_level(logging.WARNING, logger='terra.evaluation.linear')
    preds, _ = linear_regressor(
        xtr, _compositions(ytr), xva, _compositions(yva), xte, targets,
        n_epochs=5, batch_size=64, seed=0, results_save_path=str(path))
    metrics = json.loads(path.read_text())
    js = dist.jensenshannon(targets, preds, axis=-1)
    assert np.isnan(js[0])
    assert metrics['JSD'] == pytest.approx(float(np.nanmean(js)))
    assert '1 of 100 test cells have an all-zero composition' in caplog.text
