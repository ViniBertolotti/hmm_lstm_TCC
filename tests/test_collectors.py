import numpy as np
import pandas as pd
import pytest

from src.data.collectors import adjust_futures_roll, adjust_ohlc_roll, load_public_futures_close


def test_load_public_futures_close_returns_close_column_without_gaps(tmp_path):
    dates = pd.bdate_range("2020-01-02", periods=4)
    daily = pd.DataFrame({"price_index": [100.0, np.nan, 101.0, 99.5],
                          "raw_price": [500.0, np.nan, 505.0, 497.5],
                          "price_contract": ["20201100"] * 4}, index=dates)
    path = tmp_path / "SOYBEAN_daily.parquet"
    daily.to_parquet(path)

    result = load_public_futures_close(path)

    assert list(result.columns) == ["Close"]
    assert list(result.index) == [dates[0], dates[2], dates[3]]  # ausência fica para align_and_fill_ohlc
    assert result.loc[dates[3], "Close"] == 99.5


def test_large_market_loss_is_preserved_without_roll_evidence():
    raw = pd.Series([100., 75., 90.], index=pd.bdate_range('2020-01-01', periods=3))
    pd.testing.assert_series_equal(adjust_futures_roll(raw), raw)


def test_verified_roll_preserves_real_return_on_roll_day():
    dates = pd.bdate_range('2020-01-01', periods=4)
    raw = pd.Series([100., 101., 101. * .9 * 1.2, 110.], index=dates)
    adjusted = adjust_futures_roll(raw, roll_factors=pd.Series([1.2], index=[dates[2]]))
    assert adjusted.iloc[2] / adjusted.iloc[1] - 1 == pytest.approx(-.1)
    assert adjusted.iloc[-1] == raw.iloc[-1]
    assert adjusted.iloc[1] / adjusted.iloc[0] == pytest.approx(1.01)


@pytest.mark.parametrize('factor', [0., -1., np.inf, np.nan])
def test_invalid_roll_factor_rejected(factor):
    dates = pd.bdate_range('2020-01-01', periods=3)
    with pytest.raises(ValueError):
        adjust_futures_roll(pd.Series([100., 110., 120.], index=dates),
                            roll_factors=pd.Series([factor], index=[dates[1]]))


def test_ohlc_roll_preserves_geometry_volume_and_missing_dates():
    dates = pd.bdate_range('2020-01-01', periods=4)
    raw = pd.DataFrame({'Open': [100., np.nan, 110., 120.],
                        'High': [105., np.nan, 115., 125.],
                        'Low': [95., np.nan, 105., 115.],
                        'Close': [102., np.nan, 112., 122.], 'Volume': 50.}, index=dates)
    factors = pd.Series([1.1], index=[dates[2]])
    out = adjust_ohlc_roll(raw, roll_factors=factors)
    pd.testing.assert_series_equal(out.Volume, raw.Volume)
    assert out.index.equals(raw.index)
    pd.testing.assert_series_equal(out.High / out.Low, raw.High / raw.Low)
    pd.testing.assert_frame_equal(adjust_ohlc_roll(raw), raw)
