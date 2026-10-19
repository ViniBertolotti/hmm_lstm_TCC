"""Ensaio reproduzível dos quatro blocos; não usa preços ou resultados reais.

Execute da raiz: python scripts/validate_risk_pipeline.py --output reports/tables/risk_pipeline_synthetic
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

from src.data.risk_dataset import build_risk_split_dataset, make_risk_walk_forward_splits
from src.models.lstm import TrainingConfig
from src.risk.walk_forward import checkpoint_id, run_risk_fold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range('2010-01-01', periods=1700)
    rng = np.random.default_rng(20261019)
    sigma = np.where((np.arange(len(dates))//150) % 2, .025, .01)
    returns = pd.Series(rng.normal(0, sigma), index=dates)
    portfolio = pd.DataFrame({'return_1d': returns,
        'realized_vol_21d': returns.rolling(21).std()*np.sqrt(252),
        'realized_vol_63d': returns.rolling(63).std()*np.sqrt(252)})
    panel = pd.DataFrame(rng.normal(size=(len(dates), 4)), index=dates,
        columns=pd.MultiIndex.from_product([['A', 'B'], ['f1', 'f2']]))
    split = make_risk_walk_forward_splits(dates, development_years=5,
        test_months=6, calibration_sessions=252, validation_fraction=.15)[0]
    data = build_risk_split_dataset(panel, portfolio, split, sessions=dates,
        lookback=7, state_range=(2,), hmm_n_init=3, random_state=42)
    config = TrainingConfig(hidden=(8, 4), dropout=.3, max_epochs=5, patience=2, seed=42)
    outputs = {}
    for regimes in (False, True):
        result = run_risk_fold(data, training_config=config, mc_samples=1000,
            mc_seed=42, include_regimes=regimes, dm_max_lag=5)
        label = result['manifest']['model']
        frame = result['forecasts']
        assert len(frame) == 4*len(data['y_test'])
        assert (frame.origin < frame.target).all()
        assert (frame.es >= frame['var']).all()
        np.testing.assert_allclose(frame.actual_loss, -np.expm1(frame.actual_log_return))
        assert result['manifest']['effective_calibration_tail']['0.99'] < 3
        assert checkpoint_id(result['model']) == result['manifest']['model_id']
        result['forecasts'].to_parquet(out / f'{label}_forecasts.parquet')
        result['calibration'].to_parquet(out / f'{label}_calibration.parquet')
        torch.save(result['model'].state_dict(), out / f'{label}_checkpoint.pt')
        (out / f'{label}_manifest.json').write_text(json.dumps(result['manifest'],
            ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        outputs[label] = {'model_id': result['manifest']['model_id'],
            'n_sequences': result['manifest']['n_sequences'],
            'point_mse': result['manifest']['point_mse'],
            'effective_calibration_tail': result['manifest']['effective_calibration_tail']}
    for block in ('train', 'validation', 'calibration', 'test'):
        assert data[f'target_dates_{block}'].isin(getattr(split, block)).all()
    manifest = {'status': 'executed_synthetic_only', 'real_market_data_used': False,
        'data_seed': 20261019, 'development_years': 5, 'calibration_sessions': 252,
        'validation_fraction_before_calibration': .15, 'test_months': 6,
        'interpretation': 'Implementation and chronology checks; not financial performance or coverage evidence.',
        'calibration_warning': 'A 251-observation residual sample has only 2.51 observations of effective tail mass at 99%; inadequate for strong financial conclusions.',
        'models': outputs, 'R_MSGARCH_executed': False,
        'numpy': np.__version__, 'pandas': pd.__version__, 'torch': torch.__version__,
        'checks': ['four_disjoint_blocks', 'next_session_targets_inside_each_block',
                   'both_variants_trained_without_calibration_or_test',
                   '1000_date_keyed_mc_draws_per_origin', 'frozen_checkpoint',
                   'loss_units_and_es_var_order', 'tail_sample_recorded', 'no_es_pvalue_without_null_paths'],
        'source_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'):
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'src').rglob('*.py'))}}
    manifest['source_sha256'][str(Path(__file__).relative_to(ROOT)).replace('\\', '/')] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / 'pipeline_manifest.json').write_text(json.dumps(manifest,
        ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': manifest['status'], 'models': outputs}, ensure_ascii=False))


if __name__ == '__main__':
    main()
