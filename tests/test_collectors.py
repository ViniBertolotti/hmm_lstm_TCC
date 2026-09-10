import numpy as np
import pandas as pd
import pytest

from src.data.collectors import adjust_futures_roll, adjust_ohlc_roll


def _synthetic_futures_series() -> pd.Series:
    """Série sintética com retornos reais pequenos + um salto artificial de rolagem."""
    dates = pd.bdate_range("2020-01-01", periods=40)
    rng = np.random.default_rng(42)
    daily_returns = rng.normal(loc=0.0, scale=0.005, size=len(dates))

    prices = 100 * np.exp(np.cumsum(daily_returns))
    roll_idx = 20
    roll_jump = 0.15  # 15%: bem acima do jump_threshold default (8%)
    prices[roll_idx:] *= 1 + roll_jump

    return pd.Series(prices, index=dates)


def test_adjust_futures_roll_removes_artificial_jump():
    raw = _synthetic_futures_series()
    adjusted = adjust_futures_roll(raw, jump_threshold=0.08)

    adjusted_log_ret = np.log(adjusted).diff().dropna()
    assert adjusted_log_ret.abs().max() < 0.08


def test_adjust_futures_roll_preserves_percentage_returns():
    """O ajuste é multiplicativo (por razão): fora do dia exato da rolagem, os
    retornos percentuais/log ficam idênticos, já que multiplicar por uma
    constante não altera `preco[t] / preco[t-1]`. No dia da rolagem em si, o
    salto artificial é zerado de propósito (é o que a função existe para fazer)."""
    raw = _synthetic_futures_series()
    adjusted = adjust_futures_roll(raw, jump_threshold=0.08)

    roll_date = raw.index[20]
    raw_log_ret = np.log(raw).diff().dropna().drop(roll_date)
    adjusted_log_ret = np.log(adjusted).diff().dropna().drop(roll_date)

    pd.testing.assert_series_equal(raw_log_ret, adjusted_log_ret, check_names=False)


def test_adjust_futures_roll_never_produces_negative_prices_over_many_rolls():
    """Regressão: um ajuste aditivo puro, aplicado a décadas de histórico com
    muitas rolagens, empurra os preços mais antigos para valores negativos
    (achado real ao rodar em ZS=F/KC=F de 2005-2024). O ajuste multiplicativo
    não pode fazer isso, não importa quantas rolagens se acumulem."""
    dates = pd.bdate_range("2005-01-01", periods=5000)
    rng = np.random.default_rng(7)
    daily_returns = rng.normal(loc=0.0, scale=0.01, size=len(dates))
    prices = pd.Series(50 * np.exp(np.cumsum(daily_returns)), index=dates)

    # simula ~80 rolagens (a cada ~63 dias, quarterly-ish) com saltos de -20% a +20%
    for roll_idx in range(63, len(prices), 63):
        jump_factor = 1 + rng.uniform(-0.2, 0.2)
        prices.iloc[roll_idx:] *= jump_factor

    adjusted = adjust_futures_roll(prices, jump_threshold=0.08)

    assert (adjusted > 0).all()


def test_adjust_futures_roll_zeroes_the_roll_day_jump():
    raw = _synthetic_futures_series()
    adjusted = adjust_futures_roll(raw, jump_threshold=0.08)

    roll_date = raw.index[20]
    assert adjusted.diff().loc[roll_date] == pytest.approx(0.0, abs=1e-9)


def test_adjust_futures_roll_keeps_latest_price_level_unchanged():
    raw = _synthetic_futures_series()
    adjusted = adjust_futures_roll(raw, jump_threshold=0.08)

    assert adjusted.iloc[-1] == raw.iloc[-1]


def _synthetic_futures_ohlc() -> pd.DataFrame:
    close = _synthetic_futures_series()
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close},
        index=close.index,
    )


def test_adjust_ohlc_roll_close_matches_series_version():
    ohlc = _synthetic_futures_ohlc()
    adjusted_ohlc = adjust_ohlc_roll(ohlc, jump_threshold=0.08)
    adjusted_close_only = adjust_futures_roll(ohlc["Close"], jump_threshold=0.08)

    pd.testing.assert_series_equal(adjusted_ohlc["Close"], adjusted_close_only, check_names=False)


def test_adjust_ohlc_roll_keeps_high_low_consistent():
    ohlc = _synthetic_futures_ohlc()
    adjusted = adjust_ohlc_roll(ohlc, jump_threshold=0.08)

    assert (adjusted["High"] >= adjusted["Low"]).all()
    assert (adjusted["High"] >= adjusted["Close"]).all()
    assert (adjusted["Low"] <= adjusted["Close"]).all()
