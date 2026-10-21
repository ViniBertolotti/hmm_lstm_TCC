from copy import deepcopy

import numpy as np
import pandas as pd

from src.data.risk_dataset import build_risk_split_dataset, make_risk_walk_forward_splits
from src.models.lstm import TrainingConfig
from src.risk.walk_forward import run_risk_fold


def data_fixture():
    dates = pd.bdate_range('2010-01-01', periods=1500)
    rng = np.random.default_rng(16)
    returns = pd.Series(rng.normal(0, .01, len(dates)), index=dates)
    portfolio = pd.DataFrame({'return_1d': returns,
        'realized_vol_21d': returns.rolling(21).std()*np.sqrt(252),
        'realized_vol_63d': returns.rolling(63).std()*np.sqrt(252)})
    panel = pd.DataFrame(rng.normal(size=(len(dates), 2)), index=dates,
        columns=pd.MultiIndex.from_product([['A'], ['f1', 'f2']]))
    split = make_risk_walk_forward_splits(dates, calibration_sessions=40)[0]
    return build_risk_split_dataset(panel, portfolio, split, sessions=dates,
        lookback=5, state_range=(2,), hmm_n_init=2, random_state=42)


def run(data):
    return run_risk_fold(data, training_config=TrainingConfig(hidden=(4, 3),
        max_epochs=2, patience=1), mc_samples=20, include_regimes=True, dm_max_lag=2)


def test_future_inputs_or_test_targets_cannot_retrain_or_recalibrate_past():
    data = data_fixture()
    baseline = run(data)
    altered = deepcopy(data)
    altered['y_test'] += .5
    altered['X_test'][10:] += 100
    changed = run(altered)
    assert baseline['manifest']['model_id'] == changed['manifest']['model_id']
    assert baseline['manifest']['training'] == changed['manifest']['training']
    pd.testing.assert_frame_equal(baseline['calibration'], changed['calibration'])
    before = baseline['forecasts'].iloc[:40]  # first ten origins, two models, two levels
    after = changed['forecasts'].iloc[:40]
    pd.testing.assert_frame_equal(before.drop(columns=['actual_log_return', 'actual_loss']),
                                  after.drop(columns=['actual_log_return', 'actual_loss']))
    assert not before.actual_log_return.equals(after.actual_log_return)


def test_forecast_targets_units_and_empirical_tail_are_recorded_without_es_pvalue():
    data = data_fixture()
    result = run(data)
    frame, report = result['forecasts'], result['manifest']
    assert set(frame.model) == {'hmm_lstm', 'zero_mean_reference'}
    assert (frame.origin < frame.target).all()
    assert (frame.es >= frame['var']).all()
    np.testing.assert_allclose(frame.actual_loss, -np.expm1(frame.actual_log_return))
    assert report['effective_calibration_tail']['0.99'] < 1
    for model in report['es_statistics_without_null_pvalue'].values():
        for level in model.values():
            assert level['p_value_lower_tail'] is None
    assert report['training']['best_epoch'] <= 2


def test_shared_calibration_keeps_stress_labels_without_regime_pools():
    data = data_fixture()
    result = run_risk_fold(data, training_config=TrainingConfig(hidden=(4, 3),
        max_epochs=2, patience=1), mc_samples=20, include_regimes=True,
        regime_pools=False, dm_max_lag=2)
    predictions = result['forecasts'].query("model == 'hmm_lstm'")
    assert predictions.regime.notna().all()
    assert set(predictions.residual_pool) == {'all'}
    assert 'regime' not in result['calibration']
    assert 'regime_pools' not in result['manifest']['risk_method']
