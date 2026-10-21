"""Recalcula a HMM-LSTM com todos os erros, mantendo as previsões de média.

Ablação da regra de cauda: não retreina redes nem muda datas ou realizações.
Execute depois a avaliação no diretório de saída, com o mesmo MS-GARCH.
"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import file_hash, write_json  # noqa: E402
from src.risk.var_es import empirical_var_es, losses_from_log_returns  # noqa: E402


def shared_predictions(frame, calibration):
    # Os parquets históricos concatenam modelos com índices repetidos.
    result = frame.reset_index(drop=True).copy()
    selected = result.model == 'hmm_lstm'
    cal = calibration[calibration.model == 'hmm_lstm']
    residuals = (cal.return_1d - cal.prediction) / cal.scale
    for index, row in result[selected].iterrows():
        losses = losses_from_log_returns(row.prediction + row.scale * residuals.to_numpy())
        risk = empirical_var_es(losses, row.confidence)
        result.loc[index, ['var', 'es', 'residual_pool']] = [risk['var'], risk['es'], 'all']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'reports/tables/walk_forward')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.input_dir.resolve(), args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Escolha uma pasta de saída vazia para preservar as execuções.')
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    out.mkdir(parents=True, exist_ok=True)
    inputs = [source / 'manifest.json']
    for path in sorted(source.glob('forecasts_fold*.parquet')):
        cal_path = source / path.name.replace('forecasts_', 'calibration_')
        frame, calibration = pd.read_parquet(path), pd.read_parquet(cal_path)
        shared_predictions(frame, calibration).to_parquet(out / path.name)
        calibration = calibration.copy()
        # Sem regime na calibração, a reconstrução do avaliador também usa todos os erros.
        if 'regime' in calibration:
            calibration.loc[calibration.model == 'hmm_lstm', 'regime'] = float('nan')
        calibration.to_parquet(out / cal_path.name)
        inputs.extend([path, cal_path])
    for item in manifest:
        if item['model'] == 'hmm_lstm':
            item['risk_method'] = item['risk_method'].replace('_regime_pools', '')
        # Coberturas e hashes anteriores não descrevem esta transformação.
        for key in ('coverage', 'artifacts_sha256', 'es_statistics_without_null_pvalue', 'quantile_score_comparisons'):
            item.pop(key, None)
    write_json(out / 'manifest.json', manifest)
    write_json(out / 'ablation.json', {'scope': 'shared_calibration_without_retraining',
        'source': str(source), 'inputs_sha256': {p.name: file_hash(p) for p in inputs},
        'source_sha256': file_hash(Path(__file__)),
        'claim': 'Compara as redes com calibração conjunta; não é efeito causal econômico.'})
    print(f'Ablação salva em {out}')


if __name__ == '__main__':
    main()
