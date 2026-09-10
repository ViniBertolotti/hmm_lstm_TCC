import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    align_and_fill_ohlc,
    compute_return_lags,
    convert_ohlc_to_brl,
    corwin_schultz_spread,
    garch_conditional_vol_walkforward,
    garman_klass_volatility,
    get_b3_trading_days,
    equal_weighted_portfolio_return,
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


def test_get_b3_trading_days_excludes_weekends_and_new_year():
    days = get_b3_trading_days("2023-12-28", "2024-01-03")
    # 2023-12-30/31 = sab/dom, 2024-01-01 = feriado (Ano Novo) -> nenhum deve aparecer
    assert pd.Timestamp("2023-12-30") not in days
    assert pd.Timestamp("2023-12-31") not in days
    assert pd.Timestamp("2024-01-01") not in days
    assert pd.Timestamp("2023-12-28") in days
    assert pd.Timestamp("2024-01-02") in days


def test_align_and_fill_ohlc_forward_fills_short_gaps_only():
    dates = get_b3_trading_days("2021-01-04", "2021-01-20")
    close = pd.Series(range(10, 10 + len(dates)), index=dates, dtype=float)
    df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000.0})

    # remove um buraco curto (2 dias) e um buraco longo (5 dias)
    df_with_gaps = df.drop(df.index[[2, 3, 5, 6, 7, 8, 9]])

    filled = align_and_fill_ohlc(df_with_gaps, dates[0], dates[-1], max_gap_days=3)

    assert filled["Close"].iloc[2:4].notna().all()  # buraco curto: preenchido
    assert (filled["Close"].iloc[2:4] == filled["Close"].iloc[1]).all()  # ffill, não interpolação
    assert filled["Volume"].iloc[2:4].eq(0.0).all()  # sem volume nos dias "criados"
    assert filled["Close"].iloc[5:8].eq(filled["Close"].iloc[4]).all()
    assert filled["Close"].iloc[8:10].isna().all()


def test_fill_is_identical_when_future_gap_length_changes():
    dates = get_b3_trading_days("2021-01-04", "2021-01-20")
    raw = pd.DataFrame({"Close": 100.}, index=dates).drop(dates[1:6])
    full = align_and_fill_ohlc(raw, dates[0], dates[-1])
    for end in dates:
        prefix = align_and_fill_ohlc(raw.loc[:end], dates[0], end)
        pd.testing.assert_frame_equal(prefix, full.loc[:end])


def test_portfolio_aggregates_simple_returns_and_keeps_fixed_composition():
    logs = np.log1p(pd.DataFrame({"a": [.5, .1], "b": [-.5, np.nan]}))
    result = equal_weighted_portfolio_return(logs)
    assert result.iloc[0] == pytest.approx(0.)
    assert pd.isna(result.iloc[1])


def test_garch_first_forecast_matches_arch_with_nonzero_mean():
    from arch import arch_model
    returns = pd.Series(np.random.default_rng(8).normal(.02, .01, 550))
    fitted = arch_model(returns.iloc[:500] * 100, vol="Garch", p=1, q=1,
                        dist="normal", rescale=False).fit(disp="off")
    expected = np.sqrt(fitted.forecast(horizon=1).variance.iloc[-1, 0]) / 100
    actual = garch_conditional_vol_walkforward(returns)
    assert actual.iloc[500] == pytest.approx(expected, rel=1e-8)


def test_align_and_fill_ohlc_never_uses_future_prices():
    """Regressão: a versão antiga usava interpolação linear (pega o valor
    futuro conhecido para preencher o passado) — lookahead bias real. O
    ffill nunca pode preencher um buraco com um preço que só existe depois."""
    dates = get_b3_trading_days("2021-01-04", "2021-01-20")
    close = pd.Series(range(10, 10 + len(dates)), index=dates, dtype=float)
    df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000.0})
    df_with_gap = df.drop(df.index[[3]])  # um buraco de 1 dia

    filled = align_and_fill_ohlc(df_with_gap, dates[0], dates[-1], max_gap_days=3)

    # o valor preenchido tem que ser o ÚLTIMO conhecido (posição 2), nunca o próximo (posição 4)
    assert filled["Close"].iloc[3] == df["Close"].iloc[2]
    assert filled["Close"].iloc[3] != df["Close"].iloc[4]


def test_convert_ohlc_to_brl_multiplies_by_same_day_fx():
    dates = pd.bdate_range("2023-01-02", periods=5)
    ohlc_usd = pd.DataFrame(
        {"Open": 100.0, "High": 110.0, "Low": 90.0, "Close": 105.0, "Volume": 500.0}, index=dates
    )
    fx = pd.Series([5.0, 5.1, 5.2, 5.0, 4.9], index=dates)

    brl = convert_ohlc_to_brl(ohlc_usd, fx)

    pd.testing.assert_series_equal(brl["Close"], ohlc_usd["Close"] * fx, check_names=False)
    pd.testing.assert_series_equal(brl["Volume"], ohlc_usd["Volume"], check_names=False)  # volume não é convertido


def test_convert_ohlc_to_brl_preserves_high_low_ratio():
    """Garman-Klass/Corwin-Schultz usam log(High/Low) do mesmo dia — o câmbio
    do dia cancela nessa razão, então a conversão não pode alterá-la."""
    dates = pd.bdate_range("2023-01-02", periods=5)
    ohlc_usd = pd.DataFrame(
        {"Open": 100.0, "High": 110.0, "Low": 90.0, "Close": 105.0}, index=dates
    )
    fx = pd.Series([5.0, 5.1, 5.2, 5.0, 4.9], index=dates)

    brl = convert_ohlc_to_brl(ohlc_usd, fx)

    pd.testing.assert_series_equal(
        ohlc_usd["High"] / ohlc_usd["Low"], brl["High"] / brl["Low"], check_names=False
    )


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
    """Alterar um retorno em t não pode mudar a vol condicional calculada
    para qualquer instante <= t (h_t só depende de r's e h's anteriores a t —
    sem lookahead), mesmo dentro da mesma janela de refit (a recursão diária
    reage ao choque assim que ele é observado, não só no próximo refit)."""
    dates = pd.bdate_range("2015-01-01", periods=700)
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(scale=0.01, size=700), index=dates)

    cond_vol_full = garch_conditional_vol_walkforward(returns, refit_every=21, burn_in=500)

    returns_altered = returns.copy()
    returns_altered.iloc[600:] = returns_altered.iloc[600:] * 5  # choque grande a partir da posição 600
    cond_vol_altered = garch_conditional_vol_walkforward(returns_altered, refit_every=21, burn_in=500)

    # h_600 depende só de r_599 e anteriores (nenhum deles alterado) — tem que casar exatamente
    pd.testing.assert_series_equal(cond_vol_full.iloc[:601], cond_vol_altered.iloc[:601])
    # h_601 já depende de r_600 (alterado) — tem que divergir
    assert cond_vol_full.iloc[601] != cond_vol_altered.iloc[601]
