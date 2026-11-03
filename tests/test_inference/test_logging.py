"""
Test that importing the inference modules leaves logging to the application.

TERRA attaches a ``NullHandler`` to its own logger. ``terra.inference.
token_distance`` called ``logging.basicConfig(stream=sys.stdout,
level=logging.INFO)`` at import, so ``import terra`` added a handler to the
root logger and set it to INFO: the INFO records of every library were
printed, and an application's own ``logging.basicConfig`` call had no effect.
"""

import importlib.util
import logging
from pathlib import Path

import pytest

import terra.inference


@pytest.mark.parametrize('name', ['token_distance', 'embed', 'infer',
                                  'tokenize', 'harmonize', 'perturb',
                                  'score', 'summarize_distance'])
def test_module_import_does_not_configure_the_root_logger(monkeypatch, name):
    root = logging.getLogger()
    # Without handlers, as in a fresh application (pytest adds its own)
    monkeypatch.setattr(root, 'handlers', [])
    monkeypatch.setattr(root, 'level', logging.WARNING)

    # Execute a fresh copy of the module (its imports are cached)
    path = Path(terra.inference.__file__).parent / f'{name}.py'
    spec = importlib.util.spec_from_file_location(
        f'terra.inference._{name}_copy', path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))

    assert root.handlers == []
    assert root.level == logging.WARNING
