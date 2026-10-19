"""Experimento walk-forward: LSTM vs. HMM-LSTM vs. referência FHS-EWMA.

Execute da raiz:  .venv/Scripts/python scripts/run_walk_forward.py [--folds 0 1 2] [--mc-samples 1000]

Cada janela: 5 anos de desenvolvimento (treino → validação/early stopping →
252 pregões de calibração) e 6 meses de teste. Seleção de features, escala,
HMM, early stopping e calibração usam só dados anteriores ao teste. O VaR/ES
usa simulação histórica filtrada: erros da calibração padronizados pela vol
EWMA da carteira conhecida na origem. Na HMM-LSTM, a cauda vem dos erros do
regime atual quando há erros suficientes.

Saídas em reports/tables/walk_forward/: previsões por janela e um manifesto.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.features import ewma_volatility, get_b3_trading_days  # noqa: E402
from src.data.risk_dataset import build_risk_split_dataset, make_risk_walk_forward_splits  # noqa: E402
from src.models.lstm import TrainingConfig  # noqa: E402
from src.risk.walk_forward import run_risk_fold  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="*", help="janelas a rodar (padrão: todas)")
    parser.add_argument("--mc-samples", type=int, default=None)
    parser.add_argument("--min-regime-pool", type=int, default=100)
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    features = pd.read_parquet(ROOT / "data/processed/features.parquet")
    portfolio = pd.read_parquet(ROOT / "data/processed/portfolio_return.parquet")
    sessions = get_b3_trading_days(features.index.min(), features.index.max())
    features, portfolio = features.reindex(sessions), portfolio.reindex(sessions)

    scale = ewma_volatility(portfolio["return_1d"], lam=0.94)
    start = portfolio.dropna().index.min()
    splits = make_risk_walk_forward_splits(sessions[sessions >= start], calibration_sessions=252)
    chosen = args.folds if args.folds else range(len(splits))

    lstm_cfg = config["lstm"]
    training = TrainingConfig(hidden=tuple(lstm_cfg["layers"]), dropout=0.3, learning_rate=0.001,
                              batch_size=32, patience=lstm_cfg["early_stopping_patience"],
                              seed=config["random_seed"])
    mc = args.mc_samples or config["mc_dropout"]["n_samples"]
    out = ROOT / "reports/tables/walk_forward"
    out.mkdir(parents=True, exist_ok=True)

    manifests = []
    for fold in chosen:
        split = splits[fold]
        t0 = time.time()
        data = build_risk_split_dataset(features, portfolio, split, sessions=sessions, lookback=42,
                                        vif_threshold=config["features"]["vif_threshold"],
                                        hmm_n_init=10, random_state=config["random_seed"])
        frames = []
        for include in (False, True):
            result = run_risk_fold(data, training_config=training, mc_samples=mc, mc_seed=config["random_seed"],
                                   include_regimes=include, confidence_levels=tuple(config["risk"]["confidence_levels"]),
                                   scale=scale, regime_pools=True, min_regime_pool=args.min_regime_pool)
            forecasts = result["forecasts"].assign(fold=fold)
            if include:  # a referência é a mesma nos dois; guarda uma vez só
                forecasts = forecasts[forecasts.model != "zero_mean_reference"]
            frames.append(forecasts)
            manifest = result["manifest"]
            manifests.append({k: manifest[k] for k in ("fold", "model", "risk_method", "n_states", "n_sequences",
                                                       "block_dates", "coverage", "point_mse")}
                             | {"best_epoch": manifest["training"]["best_epoch"]})
        pd.concat(frames).to_parquet(out / f"forecasts_fold{fold:02d}.parquet")
        print(f"janela {fold}: teste {split.test[0].date()} -> {split.test[-1].date()} | "
              f"estados={data['n_states']} | {time.time() - t0:.0f}s", flush=True)
        (out / "manifest.json").write_text(json.dumps(manifests, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
