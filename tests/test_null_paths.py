import numpy as np
import pandas as pd
import pytest

from src.risk.null_paths import check_reconstruction, sample_scenario_losses
from src.risk.var_es import empirical_var_es, losses_from_log_returns


def test_paths_are_draws_from_each_day_scenarios_and_reproducible():
    days = [(pd.Timestamp("2020-01-02"), np.array([-.01, .0, .01])),
            (pd.Timestamp("2020-01-03"), np.array([-.05, .05]))]
    a = sample_scenario_losses(days, n_simulations=300, seed=3)
    b = sample_scenario_losses(days, n_simulations=300, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert a.shape == (300, 2)
    for (_, scenarios), column in zip(days, a.columns):
        assert set(np.round(a[column], 12)) <= set(np.round(losses_from_log_returns(scenarios), 12))


def test_reconstruction_must_match_saved_risk():
    scenarios = np.random.default_rng(0).normal(0, .01, 252)
    risk = empirical_var_es(losses_from_log_returns(scenarios), .99)
    check_reconstruction(scenarios, risk["var"], risk["es"], .99)
    with pytest.raises(ValueError, match="não reproduzem"):
        check_reconstruction(scenarios * 1.01, risk["var"], risk["es"], .99)


def test_fz0_is_minimized_by_the_true_var_and_es():
    from scipy.stats import norm
    from src.risk.backtests import fz0_score
    rng = np.random.default_rng(7)
    losses = rng.normal(0, .01, 400_000)
    c = .975
    var = norm.ppf(c) * .01
    es = .01 * norm.pdf(norm.ppf(c)) / (1 - c)
    true = fz0_score(losses, np.full_like(losses, var), np.full_like(losses, es), c).mean()
    for dv, de in [(.002, 0), (-.002, 0), (0, .003), (0, -.002), (.002, .003)]:
        other = fz0_score(losses, np.full_like(losses, var + dv), np.full_like(losses, es + de), c).mean()
        assert other > true
