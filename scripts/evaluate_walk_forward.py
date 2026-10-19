"""Avalia previsões walk-forward congeladas e registra limites da evidência.

Execute da raiz: .venv/Scripts/python scripts/evaluate_walk_forward.py
Saídas novas em reports/tables/walk_forward/evaluation_v3/. Preserva as
previsões e os relatórios anteriores; não reestima nem seleciona modelos.

Se existirem, entram também:
- o MS-GARCH (reports/tables/msgarch/forecasts_msgarch.parquet), pareado nos
  mesmos alvos;
- o p-valor do teste de ES (Acerbi-Székely Z2), com perdas sorteadas da
  distribuição que cada modelo previu. Para os modelos FHS, ela é reconstruída a
  partir dos erros da calibração (calibration_foldNN.parquet) e conferida contra
  o VaR/ES salvo antes de ser usada.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.features import get_b3_trading_days  # noqa: E402
from src.risk.calibration import fit_residual_calibrator  # noqa: E402
from src.risk.evaluation import evaluate_forecasts  # noqa: E402
from src.risk.null_paths import check_reconstruction, sample_scenario_losses  # noqa: E402

MSGARCH_DIR = ROOT / "reports/tables/msgarch"


def fhs_scenarios(frame, calibrations, manifest, *, sessions, min_regime_pool):
    """Cenários diários de log-retorno de LSTM, HMM-LSTM e referência, por alvo."""
    blocks = {(m["fold"], m["model"]): m["block_dates"] for m in manifest}
    days = {"lstm": [], "hmm_lstm": [], "zero_mean_reference": []}
    for (fold, model), cal in calibrations.groupby(["fold", "model"]):
        dates = blocks[(fold, model)]
        meta = dict(sessions=sessions, model_fit_end=dates["train"]["last"],
                    model_selection_end=dates["validation"]["last"], as_of=dates["calibration"]["last"],
                    min_regime_pool=min_regime_pool)
        cal = cal.drop(columns=["model", "fold"]).dropna(axis=1, how="all")
        fitted = {model: fit_residual_calibrator(cal, model_id=model, **meta)}
        if model == "lstm":  # a referência usa os mesmos erros com previsão zero, sem regime
            fitted["zero_mean_reference"] = fit_residual_calibrator(cal.assign(prediction=0.), model_id="ref", **meta)
        for name, calibrator in fitted.items():
            rows = frame[(frame.fold == fold) & (frame.model == name)]
            for target, day in rows.groupby("target"):
                first = day.iloc[0]
                regime = None if name != "hmm_lstm" else int(first.regime)
                pool, label = calibrator._pool(regime)
                if label != first.residual_pool:
                    raise ValueError(f"{name} {target}: cauda reconstruída difere da usada na previsão.")
                scenarios = first.prediction + first.scale * pool
                for _, row in day.iterrows():
                    check_reconstruction(scenarios, row["var"], row.es, row.confidence)
                days[name].append((target, scenarios))
    return days


def msgarch_scenarios(benchmark):
    residuals = pd.read_parquet(MSGARCH_DIR / "msgarch_residuals.parquet")
    pools = {fold: group.z.to_numpy() for fold, group in residuals.groupby("fold")}
    days = []
    for target, day in benchmark.groupby("target"):
        first = day.iloc[0]
        scenarios = first.scale * pools[first.fold]
        for _, row in day.iterrows():
            check_reconstruction(scenarios, row["var"], row.es, row.confidence)
        days.append((target, scenarios))
    return days


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT/"reports/tables/walk_forward")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dm-max-lag", type=int, default=5,
                        help="Defasagem HAC declarada; não escolher pelo menor p-valor.")
    parser.add_argument("--es-simulations", type=int, default=999)
    parser.add_argument("--min-regime-pool", type=int, default=100, help="O mesmo usado em run_walk_forward.py.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = args.input_dir.resolve()
    paths = sorted(source.glob("forecasts_fold*.parquet"))
    if not paths:
        raise ValueError("Nenhum arquivo de previsões encontrado.")
    manifest_path = source/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    sessions = get_b3_trading_days(frame.origin.min(), frame.target.max())
    inputs = paths+[manifest_path]

    benchmark = None
    msgarch_path = MSGARCH_DIR / "forecasts_msgarch.parquet"
    if msgarch_path.exists():
        benchmark = pd.read_parquet(msgarch_path)
        inputs.append(msgarch_path)

    null_losses = {}
    calibration_paths = sorted(source.glob("calibration_fold*.parquet"))
    if args.es_simulations and len(calibration_paths) == len(paths):
        calibrations = pd.concat([pd.read_parquet(p) for p in calibration_paths], ignore_index=True)
        full_sessions = get_b3_trading_days(calibrations.origin.min(), frame.target.max())
        days = fhs_scenarios(frame, calibrations, manifest, sessions=full_sessions, min_regime_pool=args.min_regime_pool)
        if benchmark is not None:
            days["ms_garch"] = msgarch_scenarios(benchmark)
        for i, (model, model_days) in enumerate(sorted(days.items())):
            null_losses[model] = sample_scenario_losses(model_days, n_simulations=args.es_simulations, seed=args.seed + i)
        inputs += calibration_paths

    result = evaluate_forecasts(frame, manifest, sessions=sessions, dm_max_lag=args.dm_max_lag,
                                benchmark=benchmark, null_losses=null_losses)
    output = (args.output_dir or source/"evaluation_v3").resolve()
    if output == source:
        raise ValueError("Escolha outra pasta de saída para preservar os relatórios anteriores.")
    output.mkdir(parents=True, exist_ok=True)
    # These hashes identify the current evaluation inputs, not the provenance
    # of the original training data, configuration or checkpoint.
    result["summary"]["evaluation_inputs_sha256"] = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p.read_bytes()).hexdigest()
                                                     for p in inputs}
    result["summary"]["evaluation_sources_sha256"] = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), ROOT/"src/risk/evaluation.py", ROOT/"src/risk/backtests.py",
                  ROOT/"src/risk/null_paths.py", ROOT/"src/data/features.py"]}
    result["summary"]["evaluation_versions"] = {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}
    result["summary"]["es_simulations"] = args.es_simulations if null_losses else 0
    result["coverage"].to_csv(output/"backtest_cobertura.csv", index=False)
    result["comparisons"].to_csv(output/"backtest_comparacao_dm.csv", index=False)
    for name, value in [("backtest_resumo.json", result["summary"]), ("backtest_detalhes.json", result["details"])]:
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "status": result["summary"]["status"],
                      "dias": result["summary"]["dias_de_teste"], "estresse": result["summary"]["dias_em_estresse"],
                      "benchmarks": result["summary"]["benchmarks"], "es_pvalue": result["summary"]["es_pvalue_models"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
