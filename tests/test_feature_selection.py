import numpy as np
import pandas as pd

from src.data.feature_selection import stationarity_report, vif_filter


def _synthetic_features(n=500, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2_collinear = x1 * 0.98 + rng.normal(scale=0.05, size=n)
    x3_independent = rng.normal(size=n)
    trend_nonstationary = np.cumsum(rng.normal(size=n))
    return pd.DataFrame(
        {
            "x1": x1,
            "x2_collinear": x2_collinear,
            "x3_independent": x3_independent,
            "trend_nonstationary": trend_nonstationary,
        }
    )


def test_vif_filter_drops_the_collinear_feature():
    df = _synthetic_features()
    kept, report = vif_filter(df, threshold=10.0)

    assert "x2_collinear" not in kept
    assert set(kept) == {"x1", "x3_independent", "trend_nonstationary"}
    assert report.loc["x2_collinear", "status"] == "dropped"
    assert report.loc["x2_collinear", "vif"] > 10.0
    assert (report.loc[kept, "vif"] <= 10.0).all()


def test_vif_filter_keeps_everything_below_threshold():
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(200, 3)), columns=["a", "b", "c"])
    kept, report = vif_filter(df, threshold=10.0)

    assert set(kept) == {"a", "b", "c"}
    assert (report["status"] == "kept").all()


def test_stationarity_report_flags_random_walk_as_nonstationary():
    df = _synthetic_features()
    report = stationarity_report(df)

    assert report.loc["trend_nonstationary", "stationary_both_tests"] == False  # noqa: E712
    assert report.loc["x1", "stationary_both_tests"] == True  # noqa: E712


def test_stationarity_report_handles_allnan_or_constant_column_gracefully():
    """Regressão: BRL=X não tem volume real no Yahoo -> volume_normalized fica
    todo NaN para esse ativo. adfuller quebra com array vazio; não pode
    derrubar o relatório inteiro por causa de uma coluna não aplicável."""
    df = _synthetic_features()
    df["all_nan"] = np.nan
    df["constant"] = 5.0

    report = stationarity_report(df)

    assert report.loc["all_nan", "stationary_both_tests"] == False  # noqa: E712
    assert report.loc["constant", "stationary_both_tests"] == False  # noqa: E712
    assert pd.isna(report.loc["all_nan", "adf_pvalue"])
    # as demais colunas continuam calculadas normalmente
    assert report.loc["x1", "stationary_both_tests"] == True  # noqa: E712
