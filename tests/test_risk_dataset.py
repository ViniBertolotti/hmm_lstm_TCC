from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.data.risk_dataset import build_risk_split_dataset, make_risk_walk_forward_splits


def synthetic_panel():
    dates = pd.bdate_range('2010-01-01', periods=1700)
    rng = np.random.default_rng(58)
    sigma = np.where((np.arange(len(dates))//120) % 2, .025, .008)
    returns = pd.Series(rng.normal(0, sigma), index=dates)
    portfolio = pd.DataFrame({'return_1d': returns,
        'realized_vol_21d': returns.rolling(21).std()*np.sqrt(252),
        'realized_vol_63d': returns.rolling(63).std()*np.sqrt(252)})
    panel = pd.DataFrame(rng.normal(size=(len(dates), 4)), index=dates,
        columns=pd.MultiIndex.from_product([['A', 'B'], ['f1', 'f2']]))
    return panel, portfolio


def build(panel, portfolio, split):
    return build_risk_split_dataset(panel, portfolio, split, sessions=panel.index,
        lookback=7, state_range=(2,), hmm_n_init=2, random_state=42)


def test_four_blocks_and_next_session_labels_are_disjoint():
    panel, portfolio = synthetic_panel()
    split = make_risk_walk_forward_splits(panel.index, calibration_sessions=252)[0]
    data = build(panel, portfolio, split)
    for name in ('train', 'validation', 'calibration', 'test'):
        dates = getattr(split, name)
        assert len(data[f'y_{name}']) > 0
        assert data[f'dates_{name}'].isin(dates).all()
        assert data[f'target_dates_{name}'].isin(dates).all()
        positions = panel.index.get_indexer(data[f'dates_{name}'])
        assert data[f'target_dates_{name}'].equals(panel.index.take(positions+1))
    assert data['target_dates_validation'].max() < data['dates_calibration'].min()
    assert data['target_dates_calibration'].max() < data['dates_test'].min()
    assert len(split.calibration) == 252


def test_calibration_and_test_values_cannot_change_training_or_selection():
    panel, portfolio = synthetic_panel()
    split = make_risk_walk_forward_splits(panel.index, calibration_sessions=80)[0]
    baseline = build(panel, portfolio, split)
    changed_panel, changed_portfolio = panel.copy(), portfolio.copy()
    changed_panel.loc[split.calibration[0]:] += 200
    changed_portfolio.loc[split.calibration[0]:, 'return_1d'] -= .5
    changed_portfolio.loc[split.calibration[0]:, ['realized_vol_21d', 'realized_vol_63d']] *= 4
    altered = build(changed_panel, changed_portfolio, split)
    assert baseline['target_std'] == altered['target_std']
    assert baseline['n_states'] == altered['n_states']
    pd.testing.assert_frame_equal(baseline['selection_report'], altered['selection_report'])
    for name in ('train', 'validation'):
        np.testing.assert_array_equal(baseline[f'X_{name}'], altered[f'X_{name}'])
        np.testing.assert_array_equal(baseline[f'y_{name}'], altered[f'y_{name}'])


def test_missing_calibration_observations_are_not_filled_or_compressed():
    panel, portfolio = synthetic_panel()
    split = make_risk_walk_forward_splits(panel.index, calibration_sessions=80)[0]
    missing = split.calibration[20]
    panel.loc[missing] = np.nan
    portfolio.loc[missing, 'return_1d'] = np.nan
    data = build(panel, portfolio, split)
    assert missing not in data['dates_calibration']
    assert missing not in data['target_dates_calibration']
    assert data['sessions'].equals(panel.index)
    for origin in data['dates_calibration']:
        position = panel.index.get_loc(origin)
        assert missing not in panel.index[position-6:position+1]


def test_omitting_calendar_sessions_and_overlapping_blocks_are_rejected():
    panel, portfolio = synthetic_panel()
    split = make_risk_walk_forward_splits(panel.index, calibration_sessions=80)[0]
    with pytest.raises(ValueError):
        build_risk_split_dataset(panel.iloc[1:], portfolio, split, sessions=panel.index)
    overlap = replace(split, calibration=split.validation)
    with pytest.raises(ValueError):
        build(panel, portfolio, overlap)


@pytest.mark.parametrize('options', [dict(calibration_sessions=1), dict(step_months=0),
    dict(development_years=True), dict(validation_fraction=1), dict(calibration_sessions=10000)])
def test_invalid_split_configuration_is_rejected(options):
    dates = pd.bdate_range('2010-01-01', periods=1700)
    with pytest.raises(ValueError):
        make_risk_walk_forward_splits(dates, **options)


def test_invalid_calendar_is_not_silently_sorted_or_deduplicated():
    dates = pd.bdate_range('2010-01-01', periods=1700)
    for bad in (dates[::-1], dates.append(dates[-1:]), dates.insert(2, pd.NaT)):
        with pytest.raises(ValueError):
            make_risk_walk_forward_splits(bad)
