"""Testes sintéticos do encadeamento, reutilizando o ajuste do repositório."""

import numpy as np
import pandas as pd

from src.data.futures_roll import build_roll_audit

from src.data.collectors import adjust_ohlc_roll


def test_numerical_oracles_and_temporal_guards():
    sessions = pd.bdate_range('2020-01-02', periods=8)
    # Contratos fictícios: o spread de 20% não deve virar retorno de mercado.
    base = np.array([100, 101, 102, 103, 104, 105, 94.5, 96.])
    rows = []
    for contract, scale in [('A', 1.), ('B', 1.2), ('C', .9)]:
        for date, price in zip(sessions, base * scale):
            rows.append(dict(date=date, contract=contract, Open=price, High=price*1.01,
                             Low=price*.99, Close=price, Volume=100))
    quotes = pd.DataFrame(rows).set_index(['date', 'contract'])
    schedule = pd.DataFrame({'roll_date': [sessions[3], sessions[6]],
        'old_contract': ['A', 'B'], 'new_contract': ['B', 'C'],
        'decision_date': [sessions[1], sessions[4]]})
    metadata = pd.DataFrame({'first_notice_date': ['2020-02-01']*3,
                            'last_trade_date': ['2020-02-15']*3}, index=['A','B','C'])

    def run(q=quotes, s=schedule, m=metadata, dates=sessions):
        return build_roll_audit(q, s, m, dates, 'A', adjust_ohlc_roll)

    result = run()
    np.testing.assert_allclose(result['returns'].iloc[1:], base[1:]/base[:-1]-1)
    assert abs(result['returns'].loc[sessions[6]] + .1) < 1e-12
    np.testing.assert_allclose(result['ledger'].factor, [1.2, .75])
    truncated = run(s=schedule.iloc[:1], dates=sessions[:6])
    pd.testing.assert_series_equal(result['causal_index'].iloc[:6], truncated['causal_index'])
    altered = quotes.copy()
    altered.loc[(sessions[-1], 'C'), ['Open','High','Low','Close']] *= 2
    pd.testing.assert_series_equal(run(q=altered)['causal_index'].iloc[:-1], result['causal_index'].iloc[:-1])
    checks = ['spread_removed_real_loss_preserved', 'same_date_factors',
              'future_roll_does_not_change_causal_history', 'future_price_does_not_change_history']

    def rejects(name, **kwargs):
        try:
            run(**kwargs)
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)

    rejects('missing_overlap_rejected', q=quotes.drop(index=(sessions[2], 'B')))
    rejects('duplicate_quote_rejected', q=pd.concat([quotes, quotes.iloc[:1]]))
    bad = schedule.copy(); bad.loc[0,'decision_date'] = sessions[3]
    rejects('future_decision_rejected', s=bad)
    bad = metadata.copy(); bad.loc['A','first_notice_date'] = str(sessions[2].date())
    rejects('late_roll_rejected', m=bad)
    bad = quotes.copy(); bad.loc[(sessions[2], 'B'), 'Close'] = 0
    rejects('nonpositive_factor_price_rejected', q=bad)
    bad = quotes.copy(); bad.loc[(sessions[4], 'B'), 'High'] = 1
    rejects('invalid_ohlc_rejected', q=bad)
