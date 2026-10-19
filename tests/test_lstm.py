"""Actual project dataset + LSTM, using simulated observations only."""
import numpy as np
import pandas as pd
from src.data.dataset import build_split_dataset, make_walk_forward_splits
from src.models.lstm import TrainingConfig, fit_lstm, predict

def test_lstm_dataset_chronology_reproducibility_and_checkpoint():
    rng = np.random.default_rng(20261019)
    dates = pd.bdate_range('2010-01-01', periods=1750)
    volatility = np.where((np.arange(len(dates)) // 150) % 2, .025, .01)
    returns = pd.Series(rng.normal(0, volatility), index=dates)
    portfolio = pd.DataFrame({'return_1d': returns,
        'realized_vol_21d': returns.rolling(21).std() * np.sqrt(252),
        'realized_vol_63d': returns.rolling(63).std() * np.sqrt(252)})
    panel = pd.DataFrame(rng.normal(size=(len(dates), 4)), index=dates,
        columns=pd.MultiIndex.from_product([['A', 'B'], ['f1', 'f2']]))
    split = make_walk_forward_splits(dates)[0]
    data = build_split_dataset(panel, portfolio, split, lookback=21,
        state_range=(2,), hmm_n_init=3, random_state=42)
    config = TrainingConfig(max_epochs=5, patience=2)
    inputs = [data[f'{kind}_{block}'] for block in ('train', 'validation')
              for kind in ('X', 'y')]
    model, record = fit_lstm(*inputs, config=config)
    again, record_again = fit_lstm(*inputs, config=config)
    p = predict(model, data['X_test'], data['target_std'])
    p2 = predict(again, data['X_test'], data['target_std'])
    np.testing.assert_array_equal(p, p2)
    assert record['history'] == record_again['history']
    np.testing.assert_array_equal(p, predict(model, data['X_test'], data['target_std']))
    best = min(record['history'], key=lambda row: row['validation_mse'])
    assert best['epoch'] == record['best_epoch']
    restored_mse = np.mean((predict(model, data['X_validation']) - data['y_validation']) ** 2)
    np.testing.assert_allclose(restored_mse, best['validation_mse'], rtol=1e-5)
    # A data do alvo precisa permanecer dentro de seu bloco cronológico.
    for block in ('train', 'validation', 'test'):
        origins = data[f'dates_{block}']
        targets = dates[dates.get_indexer(origins) + 1]
        assert targets.isin(getattr(split, block)).all()
    # Ausências devem ser rejeitadas na fronteira do treinamento.
    bad = inputs[0].copy()
    bad[0, 0, 0] = np.nan
    try:
        fit_lstm(bad, *inputs[1:], config=config)
    except ValueError:
        pass
    else:
        raise AssertionError('NaN aceito indevidamente')

    actual = data['y_test'] * data['target_std']
    origins = data['dates_test']
    forecasts = pd.DataFrame({'origin_date': origins,
        'target_date': dates[dates.get_indexer(origins) + 1],
        'prediction': p, 'actual': actual, 'zero_benchmark': 0.0})
