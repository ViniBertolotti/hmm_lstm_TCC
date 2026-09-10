import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    align_and_fill_ohlc,
    compute_return_lags,
    corwin_schultz_spread,
    garch_conditional_vol_walkforward,
    garman_klass_volatility,
)


def _synthetic_ohlc(n=60, seed=0, flat=False):
    dates = pd.bdate_range("2021-01-01", periods=n)
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(scale=0.01, size=n)))
    if flat:
        return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000}, index=dates)
    high = close * (1 + np.abs(rng.normal(scale=0.01, size=n)))
    low = close * (1 - np.abs(rng.normal(scale=0.01, size=n)))
    open_ = low + (high - low) * rng.uniform(size=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1000}, index=dates)


def test_align_and_fill_ohlc_interpolates_short_gaps_only():
    dates = pd.bdate_range("2021-01-04", periods=10)
    close = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19.0], index=dates)
    df = pd.DataFrame({"Close": close})

    # remove um buraco curto (2 dias, indices 2-3) e um buraco longo (5 dias, indices 5-9)
    df_with_gaps = df.drop(df.index[[2, 3, 5, 6, 7, 8, 9]])

    filled = align_and_fill_ohlc(df_with_gaps, dates[0], dates[-1], max_gap_days=3)

    assert filled["Close"].iloc[2:4].notna().all()  # buraco curto: interpolado
    assert filled["Close"].iloc[5:10].isna().all()  # buraco longo: continua NaN


def test_compute_return_lags_matches_manual_log_diff():
    prices = pd.Series([100, 101, 99, 105, 110.0], index=pd.bdate_range("2021-01-01", periods=5))
    out = compute_return_lags(prices, lags=(1, 2))

    expected_1d = np.log(prices).diff(1)
    expected_2d = np.log(prices).diff(2)
    pd.testing.assert_series_equal(out["return_1d"], expected_1d, check_names=False)
    pd.testing.assert_series_equal(out["return_2d"], expected_2d, check_names=False)


def test_garman_klass_volatility_is_near_zero_for_flat_ohlc():
    ohlc = _synthetic_ohlc(flat=True)
    gk = garman_klass_volatility(ohlc, windows=(21,))
    assert gk["garman_klass_vol_21d"].dropna().abs().max() < 1e-8


def test_corwin_schultz_spread_is_zero_when_high_equals_low():
    ohlc = _synthetic_ohlc(flat=True)
    spread = corwin_schultz_spread(ohlc)
    assert (spread.dropna() == 0).all()


def test_corwin_schultz_spread_is_nonnegative_on_normal_data():
    ohlc = _synthetic_ohlc(flat=False)
    spread = corwin_schultz_spread(ohlc)
    assert (spread.dropna() >= 0).all()


def test_garch_walkforward_has_nan_before_burn_in_and_values_after():
    dates = pd.bdate_range("2015-01-01", periods=700)
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(scale=0.01, size=700), index=dates)

    cond_vol = garch_conditional_vol_walkforward(returns, refit_every=21, burn_in=500)

    assert cond_vol.iloc[:500].isna().all()
    assert cond_vol.iloc[500:].notna().all()
    assert (cond_vol.dropna() > 0).all()


def test_garch_walkforward_forecast_does_not_use_future_data():
    """Alterar os retornos bem depois do primeiro refit não pode mudar a vol
    condicional já calculada antes desse ponto (sem lookahead)."""
    dates = pd.bdate_range("2015-01-01", periods=700)
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(scale=0.01, size=700), index=dates)

    cond_vol_full = garch_conditional_vol_walkforward(returns, refit_every=21, burn_in=500)

    returns_altered = returns.copy()
    returns_altered.iloc[600:] = returns_altered.iloc[600:] * 5  # choque grande só no futuro distante
    cond_vol_altered = garch_conditional_vol_walkforward(returns_altered, refit_every=21, burn_in=500)

    # refits acontecem a cada 21 dias a partir do dia 500 (500, 521, ..., 584, 605, ...);
    # o refit em 605 é o primeiro cuja janela de treino [:605] já inclui o choque em 600-604
    pd.testing.assert_series_equal(cond_vol_full.iloc[:605], cond_vol_altered.iloc[:605])
