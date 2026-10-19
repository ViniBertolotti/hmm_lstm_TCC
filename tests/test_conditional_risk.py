"""Escala de volatilidade (EWMA) e cauda por regime no cálculo do VaR/ES."""
import numpy as np
import pandas as pd
import pytest

from src.data.features import ewma_volatility
from src.risk.calibration import fit_residual_calibrator
from src.risk.walk_forward import run_risk_fold
from src.models.lstm import TrainingConfig
from tests.test_walk_forward_risk import data_fixture


def test_ewma_matches_riskmetrics_recursion():
    r = pd.Series([0.01, -0.02, 0.015, 0.0, -0.03], index=pd.bdate_range("2020-01-01", periods=5))
    vol = ewma_volatility(r, lam=0.94, warmup=2)

    var = np.mean([0.01**2, 0.02**2])
    assert np.isnan(vol.iloc[0])
    assert vol.iloc[1] == pytest.approx(np.sqrt(var))
    for t in (2, 3, 4):
        var = 0.94 * var + 0.06 * r.iloc[t] ** 2
        assert vol.iloc[t] == pytest.approx(np.sqrt(var))


def test_ewma_is_causal_and_holds_through_missing_returns():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.01, 200), index=pd.bdate_range("2020-01-01", periods=200))
    base = ewma_volatility(r)
    shocked = r.copy(); shocked.iloc[150:] *= 10
    pd.testing.assert_series_equal(base.iloc[:150], ewma_volatility(shocked).iloc[:150])

    gap = r.copy(); gap.iloc[100] = np.nan
    vol = ewma_volatility(gap)
    assert vol.iloc[100] == vol.iloc[99]


def _calibrator(regimes=None, min_pool=100, n=300):
    sessions = pd.bdate_range("2019-01-01", periods=n + 50)
    origins = sessions[20:20 + n]
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, n)
    frame = pd.DataFrame({"origin": origins, "realized_at": sessions[21:21 + n],
                          "prediction": 0.0, "return_1d": 0.01 * z, "scale": 0.01})
    if regimes is not None:
        frame["regime"] = regimes
        frame.loc[frame.regime == 1, "return_1d"] *= 4  # regime 1: caudas bem mais largas
    cal = fit_residual_calibrator(frame, sessions=sessions, model_id="m", model_fit_end=sessions[5],
                                  model_selection_end=sessions[10], as_of=sessions[20 + n], min_regime_pool=min_pool)
    return cal, sessions, 20 + n


def test_var_scales_with_volatility_known_at_origin():
    cal, sessions, pos = _calibrator()
    low = cal.forecast(0.0, origin=sessions[pos], target=sessions[pos + 1], model_id="m", scale=0.01)
    high = cal.forecast(0.0, origin=sessions[pos], target=sessions[pos + 1], model_id="m", scale=0.03)
    assert (high["var"] > 2.5 * low["var"]).all()


def test_regime_pool_used_only_with_enough_errors():
    regimes = np.r_[np.zeros(180, int), np.ones(120, int)]
    cal, sessions, pos = _calibrator(regimes, min_pool=100)
    calm = cal.forecast(0.0, origin=sessions[pos], target=sessions[pos + 1], model_id="m", scale=0.01, regime=0)
    stress = cal.forecast(0.0, origin=sessions[pos], target=sessions[pos + 1], model_id="m", scale=0.01, regime=1)
    assert (calm.residual_pool == "regime_0").all() and (stress.residual_pool == "regime_1").all()
    assert (stress["var"] > 2 * calm["var"]).all()

    sparse, sessions, pos = _calibrator(regimes, min_pool=150)
    out = sparse.forecast(0.0, origin=sessions[pos], target=sessions[pos + 1], model_id="m", scale=0.01, regime=1)
    assert (out.residual_pool == "all_fallback").all()


def test_fold_with_scale_records_method_and_uses_origin_scale():
    data = data_fixture()
    scale = pd.Series(0.01, index=data["sessions"])
    scale.loc[data["dates_test"][20:]] = 0.05  # vol sobe no meio do teste
    result = run_risk_fold(data, training_config=TrainingConfig(hidden=(4, 3), max_epochs=2, patience=1),
                           mc_samples=20, include_regimes=True, dm_max_lag=2, scale=scale)
    f = result["forecasts"]
    assert result["manifest"]["risk_method"] == "filtered_historical_simulation_vol_scaled_regime_pools"
    ref = f[(f.model == "zero_mean_reference") & (f.confidence == 0.95)].set_index("origin")["var"]
    assert ref.loc[data["dates_test"][25]] > 3 * ref.loc[data["dates_test"][5]]
