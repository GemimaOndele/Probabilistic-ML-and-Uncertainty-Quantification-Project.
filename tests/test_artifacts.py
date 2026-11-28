import numpy as np
import pytest
from experiment.artifacts import normalize_archive, read_predictions, validate_pair


def archive(tmp_path, name, **arrays):
    path = tmp_path / name
    np.savez_compressed(path, **arrays)
    return path


def test_legacy_export_roundtrip_no_overwrite(tmp_path):
    p = np.array([[[.2,.8],[.6,.4]]])
    source = archive(tmp_path, 'legacy.npz', probs_mc=p, labels=np.array([1,0]))
    target = tmp_path / 'canonical.npz'
    assert normalize_archive(source, target)['identity_verified'] is False
    data = read_predictions(target)
    np.testing.assert_array_equal(data['mc_probs'], p)
    assert 'sample_ids' not in data
    with pytest.raises(FileExistsError):
        normalize_archive(source, target)


@pytest.mark.parametrize('arrays', [
    {'probs': [[.2,.3]]}, {'probs': [[float('nan'),.5]]},
    {'probs': [[-.1,1.1]]}, {'mc_probs': [[.2,.8]]},
    {'probs': [[.2,.8]], 'logits': [[1,2]]},
    {'probs': [[.2,.8]], 'labels': [1.5]},
    {'probs': [[.2,.8]], 'labels': [2]},
    {'probs': [[.2,.8]], 'sample_ids': ['']},
])
def test_reject_invalid_archives(tmp_path, arrays):
    with pytest.raises(ValueError):
        read_predictions(archive(tmp_path, 'bad.npz', **arrays))


def test_pair_overlap_missing_ids_and_disjoint(tmp_path):
    cal = archive(tmp_path, 'cal.npz', probs=[[.2,.8]], labels=[1], sample_ids=['train:15'])
    test = archive(tmp_path, 'test.npz', probs=[[.2,.8]], labels=[1], sample_ids=['train:15'])
    with pytest.raises(ValueError, match='overlap'):
        validate_pair(cal, test)
    test = archive(tmp_path, 'test.npz', probs=[[.2,.8]], labels=[1], sample_ids=['test:15'])
    assert validate_pair(cal, test)['overlap'] == 0
    missing = archive(tmp_path, 'missing.npz', probs=[[.2,.8]], labels=[1])
    with pytest.raises(ValueError, match='sample_ids'):
        validate_pair(cal, missing)
