"""Testes de integridade e dependência temporal da reconstrução exploratória."""

import numpy as np
import pandas as pd

from src.data.public_futures_chain import reconstruct_multiple, select_daily_snapshots

from src.data.collectors import adjust_futures_roll


def test_numerical_oracles_and_temporal_guards():
    times = pd.date_range("2020-01-02 23:00", periods=6, freq="B")
    data = pd.DataFrame({"PRICE": [100, 101, 122.4, 123.6, 83.43, 84.24],
                         "PRICE_CONTRACT": ["A", "A", "B", "B", "C", "C"],
                         "FORWARD": [120, 121.2, 91.8, 92.7, 91.773, 92.664],
                         "FORWARD_CONTRACT": ["B", "B", "C", "C", "D", "D"],
                         "CARRY": [np.nan] * 6,
                         "CARRY_CONTRACT": ["Z"] * 6}, index=times)
    chain, ledger = reconstruct_multiple(data, adjust_futures_roll)
    expected = [np.nan, np.log(1.01), np.log(102/101), np.log(103/102), np.log(.9), np.log(104/103)]
    np.testing.assert_allclose(chain.log_return, expected, atol=1e-12)
    np.testing.assert_allclose(ledger.factor, [1.2, .75], atol=1e-12)
    checks = ["same_snapshot_spreads_removed_real_loss_preserved", "repository_adjustment_agrees"]
    prefix, _ = reconstruct_multiple(data.iloc[:4], adjust_futures_roll)
    pd.testing.assert_series_equal(prefix.price_index, chain.price_index.iloc[:4])
    altered = data.copy(); altered.loc[times[-1], "PRICE"] *= 1.5
    future, _ = reconstruct_multiple(altered)
    pd.testing.assert_series_equal(chain.price_index.iloc[:-1], future.price_index.iloc[:-1])
    checks += ["future_roll_does_not_change_past_index", "future_quote_does_not_change_past_index"]
    missing = data.copy(); missing.loc[times[-1], "PRICE"] = np.nan
    missing.loc[times[-1] + pd.offsets.BDay(1)] = data.iloc[-1]
    incomplete, _ = reconstruct_multiple(missing)
    assert pd.isna(incomplete.price_index.iloc[-2])
    assert incomplete.log_return.iloc[-2:].isna().all()
    assert pd.notna(incomplete.price_index.iloc[-1])
    checks.append("missing_price_and_adjacent_returns_not_filled")
    missing = data.copy(); missing.loc[times[2], "PRICE"] = np.nan
    incomplete, events = reconstruct_multiple(missing, adjust_futures_roll)
    assert incomplete.log_return.iloc[2:4].isna().all()
    assert events.price_missing_at_contract_change.iloc[0]
    np.testing.assert_allclose(incomplete.price_index.iloc[[0, 1, 3, 4, 5]],
                               chain.price_index.iloc[[0, 1, 3, 4, 5]])
    checks.append("missing_new_quote_does_not_invent_price_or_lose_verified_factor")
    missing = data.copy(); missing.loc[times[-1] + pd.Timedelta(seconds=1)] = data.iloc[-1]
    missing.iloc[-1, missing.columns.get_loc("PRICE")] = np.nan
    final, _ = reconstruct_multiple(missing)
    selected = select_daily_snapshots(final)
    assert pd.isna(selected.raw_price.iloc[-1])
    checks.append("missing_last_snapshot_not_replaced_by_earlier_quote")

    def rejects(name, frame):
        try:
            reconstruct_multiple(frame)
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)

    bad = data.copy(); bad.loc[times[1], "FORWARD_CONTRACT"] = "WRONG"
    rejects("unmatched_new_contract_rejected", bad)
    bad = data.copy(); bad.loc[times[1], "FORWARD"] = np.nan
    rejects("missing_overlap_rejected", bad)
    bad = data.copy(); bad.loc[times[1], "PRICE"] = 0
    rejects("nonpositive_price_rejected", bad)
    rejects("duplicate_timestamp_rejected", pd.concat([data, data.iloc[-1:]]))
    rejects("unsorted_timestamp_rejected", data.iloc[::-1])
    bad = data.copy(); bad.loc[times[1], ["CARRY", "CARRY_CONTRACT"]] = [120, "B"]
    rejects("conflicting_same_contract_prices_rejected", bad)
