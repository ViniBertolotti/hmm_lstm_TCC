import numpy as np
import pytest

from src.models.hmm import (
    block_bootstrap_stability,
    characterize_regimes,
    fit_gaussian_hmm,
    forward_filter,
    select_n_states,
    viterbi_decode,
    _effectively_converged,
<<<<<<< HEAD
    _BatchedEmissionHMM,
)


def test_batched_emissions_match_hmmlearn_across_independent_sequences():
    from hmmlearn.hmm import GaussianHMM
    obs, _ = _two_regime_synthetic(n=120)
    lengths = [21, 21, 21, 21, 21, 15]
    args = dict(n_components=2, covariance_type='full', n_iter=15, random_state=42)
    original = GaussianHMM(**args).fit(obs, lengths=lengths)
    batched = _BatchedEmissionHMM(**args).fit(obs, lengths=lengths)
    np.testing.assert_allclose(batched.means_, original.means_, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(batched.covars_, original.covars_, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(batched.transmat_, original.transmat_, rtol=1e-9, atol=1e-11)
    assert batched.score(obs, lengths) == pytest.approx(original.score(obs, lengths), rel=1e-10)
    assert not hasattr(batched, '_emission_cache')


=======
)


>>>>>>> 82a779c875e26e294f16e974965f241e480e5012
def test_iteration_limit_is_not_convergence():
    from types import SimpleNamespace
    for history in ([1.], [1., 3.], [3., 2.]):
        model = SimpleNamespace(tol=.01, monitor_=SimpleNamespace(history=history, converged=True))
        assert not _effectively_converged(model)
    model.monitor_.history = [3., 3.001]
    assert _effectively_converged(model)


def test_forward_resets_after_missing_period_and_handles_unreachable_state():
    from hmmlearn.hmm import GaussianHMM
    model = GaussianHMM(n_components=2, covariance_type="full")
    model.startprob_ = np.array([1., 0.])
    model.transmat_ = np.eye(2)
    model.means_ = np.array([[0., .1], [0., .3]])
    model.covars_ = np.array([np.eye(2), np.eye(2)])
    obs = np.array([[0., .1], [0., .2], [0., .3]])
    actual = forward_filter(model, obs, lengths=[1, 2])
    expected = np.vstack([forward_filter(model, obs[:1]), forward_filter(model, obs[1:])])
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual, [[1., 0.]] * 3)


def _two_regime_synthetic(n=1500, seed=42):
    """Série sintética com 2 regimes conhecidos: calmo (baixo retorno/vol) e
    estressado (retorno negativo/vol alta), com persistência realista."""
    rng = np.random.default_rng(seed)
    means = {0: [0.001, 0.10], 1: [-0.002, 0.30]}
    covs = {0: [[0.0001, 0], [0, 0.0004]], 1: [[0.0009, 0], [0, 0.0025]]}

    true_states = np.zeros(n, dtype=int)
    obs = np.zeros((n, 2))
    state = 0
    for t in range(n):
        if rng.random() < 0.02:
            state = 1 - state
        true_states[t] = state
        obs[t] = rng.multivariate_normal(means[state], covs[state])
    return obs, true_states


def test_select_n_states_picks_the_true_number_of_regimes():
    obs, _ = _two_regime_synthetic()
    report = select_n_states(obs, state_range=(2, 3, 4), n_init=5, random_state=0)

    assert report["bic"].idxmin() == 2
    assert report["aic"].idxmin() == 2


def test_viterbi_decode_recovers_true_regimes():
    obs, true_states = _two_regime_synthetic()
    model = fit_gaussian_hmm(obs, n_states=2, n_init=10, random_state=0)

    decoded = viterbi_decode(model, obs)
    agreement = max((decoded == true_states).mean(), (decoded != true_states).mean())
    assert agreement > 0.95


def test_characterize_regimes_orders_states_by_volatility():
    obs, _ = _two_regime_synthetic()
    model = fit_gaussian_hmm(obs, n_states=2, n_init=10, random_state=0)

    summary = characterize_regimes(model, obs, ["return_1d", "realized_vol_21d"])

    assert summary.loc[0, "mean_realized_vol_21d"] < summary.loc[1, "mean_realized_vol_21d"]
    assert (summary["persistence"] > 0.5).all()  # regimes persistentes, não ruído puro


def test_forward_filter_is_a_valid_probability_distribution():
    obs, _ = _two_regime_synthetic()
    model = fit_gaussian_hmm(obs, n_states=2, n_init=10, random_state=0)

    alpha = forward_filter(model, obs)
    assert alpha.shape == (len(obs), 2)
    assert np.allclose(alpha.sum(axis=1), 1.0)
    assert (alpha >= 0).all()


def test_forward_filter_recovers_true_regimes():
    obs, true_states = _two_regime_synthetic()
    model = fit_gaussian_hmm(obs, n_states=2, n_init=10, random_state=0)

    filtered_state = forward_filter(model, obs).argmax(axis=1)
    agreement = max((filtered_state == true_states).mean(), (filtered_state != true_states).mean())
    assert agreement > 0.9


def test_forward_filter_does_not_use_future_observations():
    """P(estado_t | obs_1..t) não pode mudar se só o futuro distante mudar —
    é exatamente o que separa o Forward (prospectivo) do forward-backward
    (smoothing) que o hmmlearn expõe publicamente."""
    obs, _ = _two_regime_synthetic()
    model = fit_gaussian_hmm(obs, n_states=2, n_init=10, random_state=0)

    alpha_full = forward_filter(model, obs)

    obs_altered = obs.copy()
    obs_altered[1400:] = obs_altered[1400:] + 5.0  # choque grande só no futuro distante
    alpha_altered = forward_filter(model, obs_altered)

    np.testing.assert_allclose(alpha_full[:1400], alpha_altered[:1400], atol=1e-10)


def test_block_bootstrap_stability_returns_sensible_summary():
    obs, _ = _two_regime_synthetic(n=800)
    result = block_bootstrap_stability(
        obs, state_range=(2, 3, 4), block_length=21, n_bootstrap=5, n_init=3, random_state=0
    )

    assert result["reference_n_states"] == 2
    assert result["n_bootstrap_completed"] > 0
    assert 0.0 <= result["regime_agreement_pct_mean"] <= 1.0
    assert -1.0 <= result["adjusted_rand_index_mean"] <= 1.0
    # regimes bem separados/persistentes -> bootstrap deveria concordar na maior parte do tempo
    assert result["regime_agreement_pct_mean"] > 0.8
