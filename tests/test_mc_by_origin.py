import numpy as np
import pandas as pd
import pytest
import torch

from src.models.lstm import ReturnLSTM, TrainingConfig
from src.models.mc_dropout import sample_predictions_by_origin


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    model = ReturnLSTM(3, TrainingConfig(hidden=(4, 3)))
    X = np.random.default_rng(7).normal(size=(8, 7, 3))
    dates = pd.bdate_range('2020-01-01', periods=len(X))
    return model, X, dates


def test_future_additions_missing_sequences_and_permutations_preserve_each_origin():
    model, X, dates = fixture()
    full = sample_predictions_by_origin(model, X, dates, n_samples=50)
    prefix = sample_predictions_by_origin(model, X[:3], dates[:3], n_samples=50)
    np.testing.assert_array_equal(prefix, full[:, :3])
    positions = [7, 0, 3]
    sparse = sample_predictions_by_origin(model, X[positions], dates[positions], n_samples=50)
    np.testing.assert_array_equal(sparse, full[:, positions])
    changed = X.copy(); changed[3:] *= -100
    altered = sample_predictions_by_origin(model, changed, dates, n_samples=50)
    np.testing.assert_array_equal(altered[:, :3], full[:, :3])


def test_vectorized_draws_have_dispersion_and_restore_modes_weights_and_rng():
    model, X, dates = fixture()
    model.train(); model.dropout_output.eval()
    modes = [module.training for module in model.modules()]
    state = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    draws = sample_predictions_by_origin(model, X, dates, n_samples=1000)
    assert np.all(draws.std(axis=0) > 0)
    assert all(torch.equal(model.state_dict()[key], value) for key, value in state.items())
    assert [module.training for module in model.modules()] == modes
    assert torch.equal(rng, torch.random.get_rng_state())
    np.testing.assert_array_equal(draws, sample_predictions_by_origin(model, X, dates, n_samples=1000))
    assert not np.array_equal(draws, sample_predictions_by_origin(model, X, dates, n_samples=1000, seed=8))


def test_error_also_restores_modes_and_rng():
    model, X, dates = fixture()
    model.train()
    modes = [module.training for module in model.modules()]
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        sample_predictions_by_origin(model, X[:, :, :-1], dates, n_samples=5)
    assert [module.training for module in model.modules()] == modes
    assert torch.equal(rng, torch.random.get_rng_state())


def test_duplicate_or_missing_origins_and_invalid_draw_counts_are_rejected():
    model, X, dates = fixture()
    for bad in (dates[:3], dates[:-1].append(dates[-2:-1]), dates.where(dates != dates[0])):
        with pytest.raises(ValueError):
            sample_predictions_by_origin(model, X, bad)
    for count in (0, 1, True, 3.5):
        with pytest.raises(ValueError):
            sample_predictions_by_origin(model, X, dates, n_samples=count)
