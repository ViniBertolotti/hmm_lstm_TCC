"""Dropout sampling preserves the checkpoint and respects sequence isolation."""
import numpy as np
import pytest
import torch

from src.models.lstm import ReturnLSTM, TrainingConfig
from src.models.mc_dropout import sample_predictions, summarize_predictions


def setup_model():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = ReturnLSTM(3, TrainingConfig(hidden=(4, 3), dropout=.3))
    model.train()
    model.dropout_output.eval()  # Restore mixed modes as well as uniform modes.
    X = np.random.default_rng(9).normal(size=(6, 7, 3))
    return model, X


def test_mc_reproduces_seed_without_mutating_model_or_external_rng():
    model, X = setup_model()
    states = [module.training for module in model.modules()]
    checkpoint = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    draws = sample_predictions(model, X, n_samples=1000)
    assert draws.shape == (1000, len(X))
    assert [module.training for module in model.modules()] == states
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in checkpoint.items())
    np.testing.assert_array_equal(draws, sample_predictions(model, X, n_samples=1000))
    assert not np.array_equal(draws, sample_predictions(model, X, n_samples=1000, seed=43))
    summary = summarize_predictions(draws)
    assert np.all(summary['dropout_sd'] > 0)
    np.testing.assert_allclose(summary['mc_standard_error_mean'],
                               draws.astype(np.float64).std(axis=0, ddof=1)/np.sqrt(1000))


def test_future_sequence_values_do_not_change_previous_predictions_in_same_batch():
    model, X = setup_model()
    first = sample_predictions(model, X, n_samples=20)
    changed = X.copy()
    changed[3:] += 100
    second = sample_predictions(model, changed, n_samples=20)
    np.testing.assert_array_equal(first[:, :3], second[:, :3])
    # This checks changing values in a fixed batch, not invariance to batch size.


def test_mc_restores_modes_and_rng_after_invalid_shape():
    model, X = setup_model()
    states = [module.training for module in model.modules()]
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        sample_predictions(model, X[:, :, :-1], n_samples=3)
    assert [module.training for module in model.modules()] == states
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize('count', [0, 1, True, 2.5])
def test_invalid_sample_count_rejected(count):
    model, X = setup_model()
    with pytest.raises(ValueError):
        sample_predictions(model, X, n_samples=count)


def test_scale_is_applied_once_to_mc_predictions():
    model, X = setup_model()
    plain = sample_predictions(model, X, n_samples=20)
    scaled = sample_predictions(model, X, n_samples=20, target_std=.02)
    np.testing.assert_allclose(scaled, plain*.02, rtol=1e-6)
