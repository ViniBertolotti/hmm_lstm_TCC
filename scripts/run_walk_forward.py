"""Experimento walk-forward: LSTM vs. HMM-LSTM vs. referência FHS-EWMA.

Execute da raiz:  .venv/Scripts/python scripts/run_walk_forward.py [--folds 0 1 2] [--mc-samples 1000]

Cada janela: 5 anos de desenvolvimento (treino → validação/early stopping →
252 pregões de calibração) e 6 meses de teste. Seleção de features, escala,
HMM, early stopping e calibração usam só dados anteriores ao teste. O VaR/ES
usa simulação histórica filtrada: erros da calibração padronizados pela vol
EWMA da carteira conhecida na origem. Na HMM-LSTM, a cauda vem dos erros do
regime atual quando há erros suficientes.

Saídas em reports/tables/walk_forward/: previsões por janela e um manifesto.

Robustez ativo a ativo: --target-asset SLCE3.SA troca o alvo pelo log-retorno do ativo
(em reais) e a escala pela EWMA dele. Entradas, janelas e regime do HMM continuam os
da carteira, para que o "estresse" seja o mesmo do experimento principal.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.features import ewma_volatility, get_b3_trading_days  # noqa: E402
from src.experiment import (completed_folds, file_hash, fold_artifacts, prepare_output,
                            snapshot, write_json)  # noqa: E402
from src.data.risk_dataset import build_risk_split_dataset, make_risk_walk_forward_splits  # noqa: E402
from src.models.lstm import TrainingConfig  # noqa: E402
from src.risk.walk_forward import run_risk_fold  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="*", help="janelas a rodar (padrão: todas)")
    parser.add_argument("--mc-samples", type=int, default=None)
    parser.add_argument("--min-regime-pool", type=int, default=None, help="padrão: config.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/tables/walk_forward")
    parser.add_argument("--target-asset", help="alvo univariado (ex.: KC=F); padrão: carteira")
    parser.add_argument("--resume", action="store_true",
                        help="retoma apenas com dados, código, ambiente e configuração idênticos")
    parser.add_argument('--shared-calibration', action='store_true',
                        help='ablação: HMM-LSTM também usa todos os erros, sem caudas por regime')
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    if config['hmm']['covariance_type'] != 'full' or 'bic' not in config['hmm']['selection_criteria']:
        raise ValueError('O HMM implementado usa covariância full e seleção por BIC.')
    features = pd.read_parquet(ROOT / "data/processed/features.parquet")
    portfolio = pd.read_parquet(ROOT / "data/processed/portfolio_return.parquet")
    sessions = get_b3_trading_days(features.index.min(), features.index.max())
    features, portfolio = features.reindex(sessions), portfolio.reindex(sessions)

    start = portfolio.dropna().index.min()  # janelas sempre definidas pela carteira
    hmm_observations = None
    if args.target_asset:
        hmm_observations = portfolio
        portfolio = features[[(args.target_asset, "return_1d")]].set_axis(["return_1d"], axis=1)
    risk_cfg, wf_cfg = config["var_es"], config["split"]["walk_forward"]
    min_pool = args.min_regime_pool if args.min_regime_pool is not None else risk_cfg["min_regime_pool"]
    scale = ewma_volatility(portfolio["return_1d"], lam=risk_cfg["ewma_lambda"])
    splits = make_risk_walk_forward_splits(sessions[sessions >= start], development_years=wf_cfg["development_years"],
                                           test_months=wf_cfg["test_months"], step_months=wf_cfg["step_months"],
                                           calibration_sessions=wf_cfg["calibration_sessions"],
                                           validation_fraction=wf_cfg["validation_fraction"])
    chosen = list(args.folds if args.folds else range(len(splits)))
    if len(set(chosen)) != len(chosen) or any(f < 0 or f >= len(splits) for f in chosen):
        raise ValueError('Janelas devem ser únicas e pertencer à amostra disponível.')

    lstm_cfg = config["lstm"]
    training = TrainingConfig(hidden=tuple(lstm_cfg["layers"]), dropout=lstm_cfg["dropout"],
                              learning_rate=lstm_cfg["learning_rate"], batch_size=lstm_cfg["batch_size"],
                              patience=lstm_cfg["early_stopping_patience"],
                              seed=config["random_seed"])
    mc = args.mc_samples if args.mc_samples is not None else config["mc_dropout"]["n_samples"]
    out = args.output_dir
    experiment = snapshot(ROOT, config,
        [ROOT / 'data/processed/features.parquet', ROOT / 'data/processed/portfolio_return.parquet'],
        options={'target_asset': args.target_asset, 'mc_samples': mc, 'min_regime_pool': min_pool,
                 'regime_pools': not args.shared_calibration})
    prepare_output(out, experiment, resume=args.resume)

    manifests = []
    if args.resume and (out / "manifest.json").exists():
        manifests = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        done = completed_folds(out, manifests, chosen)
        chosen = [f for f in chosen if f not in done]
        manifests = [m for m in manifests if m["fold"] not in chosen]
        print(f"retomando: {len(done)} janelas prontas, faltam {list(chosen)}", flush=True)
    for fold in chosen:
        split = splits[fold]
        t0 = time.time()
        data = build_risk_split_dataset(features, portfolio, split, sessions=sessions, lookback=lstm_cfg["lookback"],
                                        vif_threshold=config["features"]["vif_threshold"],
                                        state_range=tuple(config['hmm']['states_range']),
                                        hmm_n_init=10, random_state=config["random_seed"],
                                        hmm_observations=hmm_observations)
        frames, calibrations, fold_manifests = [], [], []
        for include in (False, True):
            result = run_risk_fold(data, training_config=training, mc_samples=mc, mc_seed=config["random_seed"],
                                   include_regimes=include, confidence_levels=tuple(config["risk"]["confidence_levels"]),
                                   scale=scale, regime_pools=not args.shared_calibration, min_regime_pool=min_pool)
            forecasts = result["forecasts"].assign(fold=fold)
            if include:  # a referência é a mesma nos dois; guarda uma vez só
                forecasts = forecasts[forecasts.model != "zero_mean_reference"]
            frames.append(forecasts)
            # Erros da calibração: permitem reconstruir a distribuição prevista (p-valor do ES).
            calibrations.append(result["calibration"].assign(model="hmm_lstm" if include else "lstm", fold=fold))
            manifest = result["manifest"]
            manifest.update(best_epoch=manifest['training']['best_epoch'],
                            target_std=data['target_std'],
                            experiment_fingerprint=experiment['fingerprint'])
            fold_manifests.append(manifest)
            checkpoint = out / f"checkpoint_{manifest['model']}_fold{fold:02d}.pt"
            temporary = checkpoint.with_suffix('.pt.tmp')
            torch.save(result['model'].state_dict(), temporary)
            temporary.replace(checkpoint)
        pd.concat(frames).to_parquet(out / f"forecasts_fold{fold:02d}.parquet")
        pd.concat(calibrations).to_parquet(out / f"calibration_fold{fold:02d}.parquet")
        hashes = {name: file_hash(out / name) for name in fold_artifacts(fold)}
        for manifest in fold_manifests:
            manifest['artifacts_sha256'] = hashes
        manifests.extend(fold_manifests)
        print(f"janela {fold}: teste {split.test[0].date()} -> {split.test[-1].date()} | "
              f"estados={data['n_states']} | {time.time() - t0:.0f}s", flush=True)
        manifests.sort(key=lambda m: (m["fold"], m["model"]))
        write_json(out / 'manifest.json', manifests)


if __name__ == "__main__":
    main()
