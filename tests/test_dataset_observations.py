import numpy as np
import pandas as pd
import pytest

from src.data.risk_dataset import build_risk_split_dataset, make_risk_walk_forward_splits


def inputs():
    dates = pd.bdate_range('2010-01-01', periods=1500)
    rng = np.random.default_rng(65)
    sigma = np.where((np.arange(len(dates))//150) % 2, .025, .01)
    returns = pd.Series(rng.normal(0, sigma), index=dates)
    portfolio = pd.DataFrame({'return_1d': returns,
        'realized_vol_21d': returns.rolling(21).std()*np.sqrt(252),
        'realized_vol_63d': returns.rolling(63).std()*np.sqrt(252)})
    panel = pd.DataFrame(rng.normal(size=(len(dates), 2)), index=dates,
        columns=pd.MultiIndex.from_product([['A'], ['f1', 'f2']]))
    split = make_risk_walk_forward_splits(dates, calibration_sessions=40)[0]
    return panel, portfolio, split


def test_lagged_hmm_observations_do_not_shift_the_return_target():
    panel, portfolio, split = inputs()
    observations = portfolio.shift(1)
    options = dict(sessions=panel.index, lookback=5, state_range=(2,),
        hmm_n_init=2, random_state=42, hmm_observations=observations)
    baseline = build_risk_split_dataset(panel, portfolio, split, **options)
    for name in ('train', 'validation', 'calibration', 'test'):
        expected = portfolio.return_1d.reindex(baseline[f'target_dates_{name}'])
        np.testing.assert_allclose(baseline[f'y_{name}']*baseline['target_std'], expected)
    changed_target = portfolio.copy()
    changed_target.loc[split.test[0]:, 'return_1d'] += .1
    changed = build_risk_split_dataset(panel, changed_target, split, **options)
    np.testing.assert_array_equal(baseline['X_test'], changed['X_test'])
    assert not np.array_equal(baseline['y_test'], changed['y_test'])


def test_hmm_observation_calendar_and_required_columns_are_validated():
    panel, portfolio, split = inputs()
    for bad in (portfolio.iloc[1:], portfolio[['return_1d']]):
        with pytest.raises(ValueError):
            build_risk_split_dataset(panel, portfolio, split, sessions=panel.index,
                hmm_observations=bad)
