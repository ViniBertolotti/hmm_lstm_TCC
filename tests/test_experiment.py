import json

import pytest

from src.experiment import completed_folds, file_hash, prepare_output, snapshot


def experiment(root, config=None):
    return snapshot(root, config or {'seed': 42}, [root / 'features.parquet'],
                    options={'mc_samples': 10}, versions={'python': 'test'})


def fixture_files(tmp_path):
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src/model.py').write_text('version = 1\n')
    (tmp_path / 'features.parquet').write_bytes(b'input')


@pytest.mark.parametrize('change', ['data', 'code', 'config'])
def test_resume_rejects_a_different_experiment(tmp_path, change):
    fixture_files(tmp_path)
    out = tmp_path / 'out'
    prepare_output(out, experiment(tmp_path), resume=False)
    config = {'seed': 42}
    if change == 'data':
        (tmp_path / 'features.parquet').write_bytes(b'changed')
    elif change == 'code':
        (tmp_path / 'src/model.py').write_text('version = 2\n')
    else:
        config['seed'] = 7
    with pytest.raises(ValueError, match='incompat'):
        prepare_output(out, experiment(tmp_path, config), resume=True)


def test_old_outputs_are_not_claimed_as_authenticated_training(tmp_path):
    fixture_files(tmp_path)
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'forecasts_fold00.parquet').write_bytes(b'old')
    with pytest.raises(ValueError, match='experiment.json'):
        prepare_output(out, experiment(tmp_path), resume=True)
    assert (out / 'forecasts_fold00.parquet').read_bytes() == b'old'


def test_resume_checks_calibration_and_checkpoints_before_skipping(tmp_path):
    out = tmp_path
    names = ['forecasts_fold00.parquet', 'calibration_fold00.parquet',
             'checkpoint_lstm_fold00.pt', 'checkpoint_hmm_lstm_fold00.pt']
    for name in names:
        (out / name).write_bytes(b'valid')
    hashes = {name: file_hash(out / name) for name in names}
    manifest = [{'fold': 0, 'model': model, 'artifacts_sha256': hashes}
                for model in ('lstm', 'hmm_lstm')]
    assert completed_folds(out, manifest, [0]) == {0}
    (out / names[1]).unlink()
    assert completed_folds(out, manifest, [0]) == set()
    (out / names[1]).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='SHA256'):
        completed_folds(out, manifest, [0])
    assert completed_folds(out, manifest[:1], [0]) == set()


def test_new_run_requires_explicit_resume_to_keep_existing_outputs(tmp_path):
    fixture_files(tmp_path)
    out = tmp_path / 'out'
    value = experiment(tmp_path)
    prepare_output(out, value, resume=False)
    (out / 'manifest.json').write_text(json.dumps([{'fold': 0}]))
    with pytest.raises(ValueError, match='resume'):
        prepare_output(out, value, resume=False)
