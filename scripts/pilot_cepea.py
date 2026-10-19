"""Ensaio técnico CEPEA separado da especificação com futuros; uma janela.

Não adota CEPEA como metodologia final. Mantém preços repetidos e lacunas.
Execute da raiz com --repo e --output; os dados originais são somente lidos.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.features import equal_weighted_portfolio_return, get_b3_trading_days
from src.data.risk_dataset import build_risk_split_dataset, make_risk_walk_forward_splits
from src.models.lstm import TrainingConfig
from src.risk.walk_forward import run_risk_fold


def load_inputs(repo):
    raw = repo/'data/raw'
    calendar = get_b3_trading_days('2005-01-01', '2024-12-31')
    equity_path = raw/'equities_fx.parquet'
    equities = pd.read_parquet(equity_path).xs('Close', axis=1, level=1)
    if equities.index.has_duplicates or equities.columns.has_duplicates:
        raise ValueError('Equities/FX has duplicate dates or assets.')
    prices = equities.reindex(calendar).copy()
    inputs = [equity_path]
    for asset, filename in [('SOJA_CEPEA', 'cepea_soja.parquet'),
                            ('CAFE_CEPEA', 'cepea_cafe_arabica.parquet')]:
        path = raw/filename
        data = pd.read_parquet(path)
        if data.index.has_duplicates or not data.index.is_monotonic_increasing:
            raise ValueError('CEPEA dates must be sorted and unique.')
        prices[asset] = data.price_brl.reindex(calendar)
        inputs.append(path)
    observed = prices.to_numpy(dtype=float)
    if np.isinf(observed).any() or (observed[np.isfinite(observed)] <= 0).any():
        raise ValueError('Observed prices must be finite and positive.')
    if prices.shape[1] != 7:
        raise ValueError('This pilot requires the seven predeclared price proxies.')
    start = max(prices[column].first_valid_index() for column in prices)
    prices = prices.loc[start:]
    returns = np.log(prices).diff()
    target = equal_weighted_portfolio_return(returns)
    portfolio = pd.DataFrame({'return_1d': target,
        'realized_vol_21d': target.rolling(21).std()*np.sqrt(252),
        'realized_vol_63d': target.rolling(63).std()*np.sqrt(252)})
    features = {}
    for asset in prices:
        log_price = np.log(prices[asset])
        for lag in (1, 5, 21):
            features[(asset, f'return_{lag}d')] = log_price.diff(lag)
        features[(asset, 'realized_vol_21d')] = returns[asset].rolling(21).std()*np.sqrt(252)
    panel = pd.DataFrame(features, index=prices.index).shift(1)
    # HMM inputs lag independently; prediction targets remain r(t+1).
    observations = portfolio.shift(1)
    hashes = {str(path.relative_to(repo)).replace('\\', '/'):
        hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    return prices, panel, portfolio, observations, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory to preserve previous attempts.')
    out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'preparing_inputs', 'scope': 'cepea_separate_technical_pilot_not_final',
        'real_data': True, 'futures_version_replaced': False, 'original_data_changed': False,
        'selection_rule': 'First complete five-year development window and following six-month test; no outcome-based window selection.',
        'development_years': 5, 'calibration_sessions': 252,
        'validation_fraction_before_calibration': .15, 'test_months': 6,
        'lookback': 21, 'hmm_state_candidates': [2, 3, 4], 'hmm_n_init': 3,
        'mc_samples_per_origin': 1000, 'feature_lag_b3_sessions': 1,
        'primary_missing_policy': 'no filling or weight renormalization',
        'source_availability_status': 'One-session lag is a conservative assumption, not proof of historical publication times or unrevised point-in-time data.',
        'loss': 'negative_simple_return', 'horizon_sessions': 1,
        'limitations': [
            'Physical CEPEA indicators change the economic exposure relative to futures.',
            'The equal-weight price proxy is not a directly tradable physical commodity portfolio.',
            'Published constant-price runs are retained and can affect risk estimates.',
            'Only one chronologically preselected window; no final H1-H3 conclusions.',
            'Constant-scale frozen empirical residuals may not adapt to changing volatility.',
            'An approximately one-year calibration sample has very little 99% tail information.',
            'No MSGARCH run and no justified full null simulation for an ES p-value.',
            'Close-only variables omit OHLCV-based features and macro variables in this pilot.',
        ]}

    def save():
        (out/'pilot_manifest.json').write_text(json.dumps(report, ensure_ascii=False,
            indent=2, allow_nan=False), encoding='utf-8')

    save()
    try:
        prices, panel, portfolio, observations, hashes = load_inputs(args.repo.resolve())
        report['input_sha256'] = hashes
        calendar = prices.index
        first_test_end = calendar[0]+pd.DateOffset(years=5, months=6)
        eligible_calendar = calendar[calendar < first_test_end]
        split = make_risk_walk_forward_splits(eligible_calendar, development_years=5,
            test_months=6, calibration_sessions=252, validation_fraction=.15)[0]
        report.update(status='building_dataset', common_price_start=str(calendar[0].date()),
            test_block_start=str(split.test[0].date()), test_block_end=str(split.test[-1].date()))
        save()
        print(json.dumps({'stage': report['status'], 'test_start': report['test_block_start'],
            'test_end': report['test_block_end']}), flush=True)
        data = build_risk_split_dataset(panel, portfolio, split, sessions=calendar,
            lookback=21, state_range=(2, 3, 4), hmm_n_init=3, random_state=42,
            hmm_observations=observations)
        config = TrainingConfig(hidden=(64, 32), dropout=.3, learning_rate=.001,
            batch_size=32, max_epochs=20, patience=3, seed=42)
        summaries = {}
        for regimes in (False, True):
            name = 'hmm_lstm' if regimes else 'lstm'
            report['status'] = 'running_'+name
            save()
            print(json.dumps({'stage': report['status'],
                'counts': {block: len(data[f'y_{block}']) for block in ('train','validation','calibration','test')}}), flush=True)
            result = run_risk_fold(data, training_config=config, mc_samples=1000,
                mc_seed=42, include_regimes=regimes, dm_max_lag=5)
            result['manifest']['experiment_scope'] = report['scope']
            result['forecasts'].to_parquet(out/f'{name}_forecasts.parquet')
            result['calibration'].to_parquet(out/f'{name}_calibration.parquet')
            torch.save(result['model'].state_dict(), out/f'{name}_checkpoint.pt')
            (out/f'{name}_manifest.json').write_text(json.dumps(result['manifest'],
                ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            summaries[name] = {key: result['manifest'][key] for key in
                ('model_id', 'training', 'n_sequences', 'effective_calibration_tail',
                 'point_mse', 'coverage', 'quantile_score_comparisons')}
            report['completed_models'] = list(summaries)
            save()
        test_prices = prices.reindex(split.test)
        soy_returns = np.log(prices.SOJA_CEPEA).diff().reindex(split.test)
        report['soy_test_diagnostic'] = {'n_block_sessions': len(split.test),
            'n_valid_returns': int(soy_returns.notna().sum()),
            'n_zero_returns': int(soy_returns.eq(0).sum()),
            'price_min': float(test_prices.SOJA_CEPEA.min()),
            'price_max': float(test_prices.SOJA_CEPEA.max())}
        report['models'] = summaries
        report['test_coverage'] = {'block_sessions': len(split.test),
            'retained_next_session_targets': len(data['target_dates_test']),
            'retained_fraction': len(data['target_dates_test'])/len(split.test),
            'reason': 'Missing values, rolling-window requirements, HMM observations and split-boundary purge; no outcome-based exclusions.'}
        report['source_sha256'] = {str(path.relative_to(ROOT)).replace('\\', '/'):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT/'src').rglob('*.py')}
        report['source_sha256']['scripts/pilot_cepea.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report['numpy_version'], report['pandas_version'], report['torch_version'] = np.__version__, pd.__version__, torch.__version__
        report['status'] = 'executed_one_fold_not_final'
        save()
        print(json.dumps({'status': report['status'], 'test_coverage': report['test_coverage'],
            'soy': report['soy_test_diagnostic'], 'mse': {key: value['point_mse'] for key, value in summaries.items()}}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
