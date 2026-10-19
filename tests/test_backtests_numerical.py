"""Oráculos analíticos, HAC independente e cenários simulados dos backtests."""

import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_1samp
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

from src.risk.backtests import (acerbi_szekely_unconditional, diebold_mariano,
                       quantile_score, var_coverage)


def test_numerical_oracles_and_temporal_guards():
    checks = []
    dates = pd.bdate_range("2020-01-02", periods=8)
    breaches = pd.Series([0, 0, 1, 0, 1, 1, 0, 0], index=dates, dtype=float)
    var = pd.Series(.5, index=dates)
    report = var_coverage(breaches, var, .8, sessions=dates)
    assert report["transition_counts"] == dict(n00=2, n01=2, n10=2, n11=1)
    np.testing.assert_allclose([report["kupiec"]["statistic"], report["independence"]["statistic"],
                               report["conditional_coverage"]["statistic"]],
                              [1.3030511772189843, .1964510115581648, 1.4995021887771491], atol=1e-12)
    checks.append("known_likelihood_ratio_and_transition_fixture")
    for value in (0., 1.):
        edge = var_coverage(pd.Series(value, index=dates), var, .99, sessions=dates)
        assert np.isfinite(edge["kupiec"]["statistic"])
        assert edge["independence"]["status"] == "not_identifiable"
        assert edge["independence"]["p_value_chi2_1"] is None
    checks.append("zero_or_all_failures_do_not_fake_independence_acceptance")
    missing_dates = dates[:5].delete(2)
    gaps = var_coverage(pd.Series([0, 1, 1, 0], index=missing_dates),
                        pd.Series(.5, index=missing_dates), .8, sessions=dates[:5])
    assert gaps["transition_counts"] == dict(n00=0, n01=1, n10=1, n11=0)
    assert gaps["n_missing_calendar_sessions"] == 1
    checks.append("missing_session_does_not_create_false_failure_transition")
    tied = var_coverage(var, var, .95, sessions=dates)
    assert tied["n_failures"] == 0 and tied["n_observed_ties_at_var"] == 8
    checks.append("strict_breach_convention_and_ties_reported")

    dates_dm = pd.bdate_range("2021-01-01", periods=60)
    rng = np.random.default_rng(20261021)
    d = .3+rng.normal(size=len(dates_dm))
    first, second = pd.Series(5+d, index=dates_dm), pd.Series(5., index=dates_dm)
    dm = diebold_mariano(first, second, sessions=dates_dm, max_lag=0, small_sample_correction=True)
    oracle = ttest_1samp(d, 0)
    np.testing.assert_allclose([dm["statistic"], dm["p_value_two_sided"]],
                               [oracle.statistic, oracle.pvalue], rtol=1e-12)
    checks.append("hln_h1_lag0_matches_independent_one_sample_t_oracle")
    full_dm = diebold_mariano(first, second, sessions=dates_dm, max_lag=4)
    regression = sm.OLS(d, np.ones((len(d), 1))).fit()
    oracle_variance = float(cov_hac(regression, nlags=4, use_correction=False)[0, 0])
    np.testing.assert_allclose(full_dm["long_run_variance"]/len(d), oracle_variance, rtol=1e-12)
    checks.append("bartlett_hac_matches_statsmodels_independent_implementation")
    sparse_first = first.drop(dates_dm[[10, 11, 30]])
    sparse_second = second.reindex(sparse_first.index)
    sparse_dm = diebold_mariano(sparse_first, sparse_second, sessions=dates_dm, max_lag=4)
    mask = dates_dm.isin(sparse_first.index).astype(float)
    regression = sm.OLS(mask*d, mask[:, None]).fit()
    oracle_variance = float(cov_hac(regression, nlags=4, use_correction=False)[0, 0])
    np.testing.assert_allclose(sparse_dm["long_run_variance"]/int(mask.sum()), oracle_variance, rtol=1e-12)
    checks.append("calendar_gaps_preserved_in_hac_against_weighted_regression_oracle")
    reversed_dm = diebold_mariano(second, first, sessions=dates_dm, max_lag=4)
    scaled_dm = diebold_mariano(first*100, second*100, sessions=dates_dm, max_lag=4)
    np.testing.assert_allclose([reversed_dm["statistic"], reversed_dm["p_value_two_sided"]],
                               [-full_dm["statistic"], full_dm["p_value_two_sided"]], rtol=1e-12)
    np.testing.assert_allclose(scaled_dm["statistic"], full_dm["statistic"], rtol=1e-12)
    checks.append("dm_sign_and_unit_scaling")
    for a, b in [(first, first), (pd.Series(.1, index=dates_dm), second*0)]:
        zero_variance = diebold_mariano(a, b, sessions=dates_dm, max_lag=2)
        assert zero_variance["status"] == "degenerate_variance" and zero_variance["p_value_two_sided"] is None
    checks.append("constant_loss_differential_does_not_invent_p_value")
    values = np.array([-2., -1., 0., 1., 2.])
    proper = quantile_score(values, np.full(5, 1.), .8).mean()
    assert proper < quantile_score(values, np.full(5, -2.), .8).mean()
    assert proper < quantile_score(values, np.full(5, 10.), .8).mean()
    checks.append("pinball_score_favors_empirical_quantile")

    as_fixture = acerbi_szekely_unconditional([.1, 0, .2, 0], [.05]*4, [.1]*4, .5)
    np.testing.assert_allclose(as_fixture["statistic_z_unconditional"], -.5)
    assert as_fixture["p_value_lower_tail"] is None
    checks.append("acerbi_szekely_sign_and_no_p_value_without_null_distribution")
    rank_fixture = acerbi_szekely_unconditional([.1, 0, .2, 0], [.05]*4, [.1]*4, .5,
        null_paths={"losses": np.zeros((99, 4)), "var": np.full((99, 4), .05), "es": np.full((99, 4), .1)})
    np.testing.assert_allclose(rank_fixture["p_value_lower_tail"], .01)
    checks.append("simulation_rank_plus_one_and_minimum_p_value")

    def rejects(name, operation):
        try:
            operation()
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)

    rejects("unmatched_target_dates_rejected", lambda: var_coverage(breaches, var.iloc[:-1], .95, sessions=dates))
    rejects("duplicate_target_dates_rejected", lambda: var_coverage(
        pd.concat([breaches, breaches.iloc[-1:]]), pd.concat([var, var.iloc[-1:]]), .95, sessions=dates))
    rejects("infinite_loss_rejected", lambda: var_coverage(breaches.replace(1, np.inf), var, .95, sessions=dates))
    rejects("invalid_confidence_rejected", lambda: var_coverage(breaches, var, 1, sessions=dates))
    rejects("hln_does_not_compress_missing_sessions", lambda: diebold_mariano(
        sparse_first, sparse_second, sessions=dates_dm, max_lag=1, small_sample_correction=True))
    rejects("invalid_hac_bandwidth_rejected", lambda: diebold_mariano(first, second, sessions=dates_dm, max_lag=60))
    rejects("nonpositive_es_rejected_for_normalized_test", lambda: acerbi_szekely_unconditional([0, 1], [0, 0], [0, 0], .95))
    rejects("null_path_forecast_shapes_must_match", lambda: acerbi_szekely_unconditional(
        [.1, 0], [.05, .05], [.1, .1], .95, null_paths={"losses": np.zeros((99, 2)), "var": np.zeros(2), "es": np.ones((99, 2))}))
    rejects("null_es_cannot_be_below_null_var", lambda: acerbi_szekely_unconditional(
        [.1, 0], [.05, .05], [.1, .1], .95, null_paths={"losses": np.zeros((99, 2)), "var": np.ones((99, 2)), "es": np.full((99, 2), .5)}))

    # Demonstração de potência contra erro proposital, com modelo nulo conhecido.
    # Gaussianas para perdas são um exemplo matemático; não são resultados do TCC.
    n, b, sigma = 1000, 999, .01
    toy_dates = pd.bdate_range("2010-01-01", periods=n)
    observed = np.random.default_rng(20261021).normal(0, sigma, n)
    null_standard = np.random.default_rng(20261022).normal(size=(b, n))
    toy_results = {}
    for confidence in (.95, .99):
        z = norm.ppf(confidence)
        base_var = sigma*z
        base_es = sigma*norm.pdf(z)/(1-confidence)
        for factor in (1., .6):
            v, e = factor*base_var, factor*base_es
            result = acerbi_szekely_unconditional(observed, np.full(n, v), np.full(n, e), confidence,
                null_paths={"losses": null_standard*(sigma*factor),
                            "var": np.full((b, n), v), "es": np.full((b, n), e)})
            coverage = var_coverage(pd.Series(observed, index=toy_dates), pd.Series(v, index=toy_dates), confidence, sessions=toy_dates)
            toy_results[f"c{confidence}_risk_scale{factor}"] = {"es_test": result, "var_tests": coverage}
            if factor == .6:
                assert result["p_value_lower_tail"] <= .01
    checks.append("gross_risk_understatement_detected_in_seeded_normal_loss_toy_case")
