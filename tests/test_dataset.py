import numpy as np
import pandas as pd
import pytest

from src.data.dataset import (
    Scaler,
    build_split_dataset,
    causal_regime_features,
    make_sequences,
    make_walk_forward_splits,
    market_level_panel,
    select_split_features,
)


def _synthetic_portfolio(n=1800, seed=0):
    """Retorno + vol. realizada com dois regimes persistentes."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n)
    state, vol = 0, np.empty(n)
    for t in range(n):
        if rng.random() < 0.01:
            state = 1 - state
        vol[t] = 0.01 if state == 0 else 0.03
    ret = rng.normal(0, vol)
    s = pd.Series(ret, index=dates)
    return pd.DataFrame({
        "return_1d": s,
        "realized_vol_21d": s.rolling(21).std() * np.sqrt(252),
        "realized_vol_63d": s.rolling(63).std() * np.sqrt(252),
    })


def _synthetic_panel(dates, seed=1):
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_product([["A", "B"], ["f1", "f2", "f3", "implied_vol_us"]])
    panel = pd.DataFrame(rng.normal(size=(len(dates), len(cols))), index=dates, columns=cols)
    panel[("B", "implied_vol_us")] = panel[("A", "implied_vol_us")]  # variável de mercado repetida
    return panel


def test_splits_are_chronological_and_never_overlap():
    dates = pd.bdate_range("2008-01-01", "2016-12-31")
    splits = make_walk_forward_splits(dates, train_years=5, test_months=6, step_months=6)

    assert len(splits) > 1
    for s in splits:
        assert s.train.max() < s.validation.min() <= s.validation.max() < s.test.min()
        assert len(s.validation) == pytest.approx(0.15 * (len(s.train) + len(s.validation)), abs=1)
    first_tests = [s.test.min() for s in splits]
    assert all(b - a >= pd.Timedelta(days=170) for a, b in zip(first_tests, first_tests[1:]))


def test_sequences_end_at_t_and_target_is_t_plus_one():
    dates = pd.bdate_range("2020-01-01", periods=10)
    X = pd.DataFrame({"x": np.arange(10.0)}, index=dates)
    y = pd.Series(np.arange(10.0) * 10, index=dates)

    windows, targets, sample_dates = make_sequences(X, y, dates, lookback=3)

    assert sample_dates[0] == dates[2]  # primeira janela completa
    np.testing.assert_array_equal(windows[0, :, 0], [0, 1, 2])
    assert targets[0] == 30  # alvo de t+1
    assert sample_dates[-1] == dates[8]  # último dia não tem t+1


def test_sequences_skip_windows_with_missing_rows():
    dates = pd.bdate_range("2020-01-01", periods=10)
    X = pd.DataFrame({"x": np.arange(10.0)}, index=dates)
    X.iloc[4] = np.nan
    y = pd.Series(1.0, index=dates)

    _, _, sample_dates = make_sequences(X, y, dates, lookback=3)

    assert not set(dates[4:7]) & set(sample_dates)  # qualquer janela que contenha o dia 4 sai


def test_feature_selection_ignores_test_period():
    dates = pd.bdate_range("2015-01-01", periods=400)
    panel = market_level_panel(_synthetic_panel(dates))
    train = dates[:300]

    cols_a, _ = select_split_features(panel, train)
    altered = panel.copy()
    altered.iloc[300:] = np.nan  # destruir o "teste" não pode mudar a escolha
    cols_b, _ = select_split_features(altered, train)

    assert cols_a == cols_b
    assert ("market", "implied_vol_us") in cols_a
    assert ("A", "implied_vol_us") not in cols_a  # variável de mercado entra uma vez só


def test_feature_with_mostly_missing_train_is_dropped():
    dates = pd.bdate_range("2015-01-01", periods=400)
    panel = market_level_panel(_synthetic_panel(dates))
    panel.loc[dates[:250], ("A", "f1")] = np.nan  # série que só "começa" no fim do treino

    cols, report = select_split_features(panel, dates[:300], max_missing=0.2)

    assert ("A", "f1") not in cols
    assert not report.loc[("A", "f1"), "selected"]


def test_scaler_uses_train_statistics_only():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    scaler = Scaler.fit(train)
    out = scaler.transform(pd.DataFrame({"x": [100.0]}))
    assert out.iloc[0, 0] == pytest.approx((100 - 2) / 1)


def test_regimes_do_not_depend_on_test_data():
    portfolio = _synthetic_portfolio()
    dates = portfolio.index
    train, until = dates[:1200], dates[-1]

    regimes, n_states = causal_regime_features(portfolio, train, until, state_range=(2,), n_init=3, random_state=0)
    altered = portfolio.copy()
    altered.iloc[1500:] *= 10  # choque só depois do ponto 1500
    regimes_alt, _ = causal_regime_features(altered, train, until, state_range=(2,), n_init=3, random_state=0)

    assert n_states == 2
    pd.testing.assert_frame_equal(regimes.iloc[:1500], regimes_alt.iloc[:1500])
    assert np.allclose(regimes.dropna().sum(axis=1), 1.0)


def test_build_split_dataset_end_to_end():
    portfolio = _synthetic_portfolio(n=1800)
    dates = portfolio.index
    panel = _synthetic_panel(dates)
    split = make_walk_forward_splits(dates, train_years=5, test_months=6)[0]

    data = build_split_dataset(panel, portfolio, split, lookback=21, state_range=(2,), hmm_n_init=3, random_state=0)

    for name in ("train", "validation", "test"):
        assert data[f"X_{name}"].ndim == 3
        assert len(data[f"X_{name}"]) == len(data[f"y_{name}"]) > 0
        assert np.isfinite(data[f"X_{name}"]).all()
    assert data["dates_train"].max() < data["dates_validation"].min()
    assert data["dates_validation"].max() < data["dates_test"].min()
    assert any(c[0] == "regime" for c in data["columns"])
