"""
Tests for ``terra.utils.filter_dataset_by_cell_ids``.

The filtered dataset was cached under one fixed file name in `temp_dir`. With
``num_proc > 1`` the shard files of that cache were never removed, so a later
call in the same directory (e.g. a second split in the same folder, or a
re-run of the supervised finetuning tutorial with other splits) silently
returned the subset of the earlier call. Every call now writes to a cache of
its own, never reads an earlier one, and all its files are removed when the
Python session ends; the returned dataset stays usable until then (it reads
from these files, also in worker processes).
"""

import atexit
import pickle

import pytest
from datasets import Dataset

from terra.utils import filter_dataset_by_cell_ids

CELL_IDS = [f'c{i}' for i in range(40)]


@pytest.mark.parametrize('num_proc', [1, 2])
def test_repeated_calls_return_their_own_subsets(
        tmp_path, monkeypatch, num_proc):
    cleanups = []
    monkeypatch.setattr(atexit, 'register', lambda function, *args, **kwargs:
                        cleanups.append((function, args, kwargs)))
    dataset = Dataset.from_dict({'cell_id': CELL_IDS, 'x': list(range(40))})
    subsets = {'even': CELL_IDS[0::2], 'odd': CELL_IDS[1::2]}
    filtered = {
        name: filter_dataset_by_cell_ids(
            dataset, ids, num_proc=num_proc, temp_dir=tmp_path)
        for name, ids in subsets.items()}
    # Again, in the same directory
    filtered['odd_again'] = filter_dataset_by_cell_ids(
        dataset, subsets['odd'], num_proc=num_proc, temp_dir=tmp_path)

    assert filtered['even']['cell_id'] == subsets['even']
    assert filtered['odd']['cell_id'] == subsets['odd']
    assert filtered['odd_again']['cell_id'] == subsets['odd']
    # Usable in other processes (pickled by the paths of its files)
    assert (pickle.loads(pickle.dumps(filtered['odd']))['cell_id']
            == subsets['odd'])

    # At the end of the session every cache file, shards included, is removed
    # (other libraries may register exit handlers too)
    cleanups = [(function, args, kwargs)
                for function, args, kwargs in cleanups
                if args and str(args[0]).startswith(str(tmp_path))]
    assert len(cleanups) == 3
    assert list(tmp_path.rglob('*.arrow'))
    for function, args, kwargs in cleanups:
        function(*args, **kwargs)
    assert not list(tmp_path.rglob('*.arrow'))
