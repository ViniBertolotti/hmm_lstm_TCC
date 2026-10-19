"""Valida cálculo, caudas discretas e fronteiras de informação; sem dados reais."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.risk.calibration import fit_residual_calibrator
from src.risk.var_es import empirical_var_es, losses_from_log_returns


def test_numerical_oracles_and_temporal_guards():
    checks = []
    result = empirical_var_es([-.05, 0, .01, .02, .04], .7)
    np.testing.assert_allclose([result["var"], result["es"]], [.02, 1/30], atol=1e-12)
    checks.append("fractional_mass_at_empirical_quantile")
    result = empirical_var_es([0, 1, 1, 1, 3], .5)
    np.testing.assert_allclose([result["var"], result["es"]], [1, 1.8], atol=1e-12)
    checks.append("ties_at_quantile_do_not_bias_tail_mean")
    assert empirical_var_es([-.03, -.02, -.01], .5)["es"] < 0
    checks.append("gains_not_clipped_to_zero")
    np.testing.assert_allclose(losses_from_log_returns(np.log1p([-.1, .1])), [.1, -.1])
    checks.append("log_return_converted_to_fractional_capital_loss")
    grid = (np.arange(20000)+.5)/20000
    mean, sigma = .0005, .01
    scenarios = mean + sigma * norm.ppf(grid)
    oracle = []
    for level in (.95, .99):
        z = norm.ppf(1-level)
        expected_var = 1-np.exp(mean+sigma*z)
        expected_es = 1-np.exp(mean+sigma*sigma/2)*norm.cdf(z-sigma)/(1-level)
        actual = empirical_var_es(losses_from_log_returns(scenarios), level)
        np.testing.assert_allclose([actual["var"], actual["es"]],
                                   [expected_var, expected_es], rtol=0, atol=2e-5)
        assert actual["es"] >= actual["var"]
        oracle.append({"confidence": level, "empirical": actual,
                       "normal_log_return_oracle": {"var": expected_var, "es": expected_es}})
    checks.append("normal_log_return_closed_form_oracle_95_99")
    sessions = pd.bdate_range("2020-01-01 17:00", periods=16)
    frame = pd.DataFrame({"origin": sessions[2:10], "realized_at": sessions[3:11],
                          "prediction": np.full(8, .001),
                          "return_1d": [.01, -.02, 0, .015, -.03, .005, -.005, .02],
                          "scale": np.full(8, .01)})
    settings = dict(sessions=sessions, model_id="frozen-model", model_fit_end=sessions[0],
                    model_selection_end=sessions[1], as_of=sessions[10])
    model = fit_residual_calibrator(frame, **settings)
    args = dict(prediction=.001, origin=sessions[10], target=sessions[11], model_id="frozen-model", scale=.01)
    baseline = model.forecast(**args)
    pd.testing.assert_frame_equal(model.forecast(**args), baseline)
    assert (baseline.es >= baseline["var"]).all()
    assert baseline["var"].is_monotonic_increasing and baseline.es.is_monotonic_increasing
    checks += ["reproducible_empirical_distribution", "var_es_order_and_confidence_monotonicity"]
    original_frame = frame.copy()
    frame.loc[:, "return_1d"] += 1
    pd.testing.assert_frame_equal(model.forecast(**args), baseline)
    frame = original_frame
    checks.append("calibration_snapshot_owns_immutable_residuals")
    twice = frame.copy(); twice["scale"] *= 2
    twice_model = fit_residual_calibrator(twice, **settings)
    pd.testing.assert_frame_equal(twice_model.forecast(**dict(args, scale=.02)), baseline)
    checks.append("calibration_and_forecast_scale_units_agree")
    later = dict(args, prediction=-.2, origin=sessions[12], target=sessions[13])
    model.forecast(**later)
    pd.testing.assert_frame_equal(model.forecast(**args), baseline)
    checks.append("future_forecasts_do_not_revise_past_risk")

    def rejects(name, operation):
        try:
            operation()
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)

    rejects("selection_overlaps_calibration_rejected", lambda: fit_residual_calibrator(
        frame, **dict(settings, model_selection_end=sessions[5])))
    rejects("unobserved_calibration_return_rejected", lambda: fit_residual_calibrator(
        frame, **dict(settings, as_of=sessions[9])))
    bad = frame.copy(); bad.loc[0, "realized_at"] = sessions[4]
    rejects("wrong_target_session_rejected", lambda: fit_residual_calibrator(bad, **settings))
    rejects("past_origin_cannot_use_later_calibration", lambda: model.forecast(
        **dict(args, origin=sessions[9], target=sessions[10])))
    rejects("different_checkpoint_rejected", lambda: model.forecast(**dict(args, model_id="other")))
    rejects("multi_session_horizon_rejected", lambda: model.forecast(**args, horizon=5))
    bad = frame.copy(); bad.loc[0, "prediction"] = np.nan
    rejects("missing_calibration_prediction_rejected", lambda: fit_residual_calibrator(bad, **settings))
    bad = frame.copy(); bad.loc[0, "scale"] = 0
    rejects("nonpositive_calibration_scale_rejected", lambda: fit_residual_calibrator(bad, **settings))
    rejects("dropout_draw_matrix_is_not_a_loss_vector", lambda: empirical_var_es(np.ones((1000, 8)), .99))
    rejects("invalid_confidence_rejected", lambda: empirical_var_es([1, 2], 1))
    rejects("missing_loss_rejected", lambda: empirical_var_es([0, np.nan], .95))
