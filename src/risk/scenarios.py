"""Reconstrói os cenários diários de log-retorno de cada modelo.

Os modelos FHS (LSTM, HMM-LSTM e referência de média zero) são refeitos a partir dos erros
da calibração de cada janela; o MS-GARCH, a partir da vol prevista e dos resíduos
padronizados do desenvolvimento. Cada reconstrução é conferida contra o VaR/ES salvo antes
de ser usada (p-valor do ES, PIT e CRPS).
"""
from pathlib import Path

import pandas as pd

from src.risk.calibration import fit_residual_calibrator
from src.risk.null_paths import check_reconstruction

MSGARCH_DIR = Path(__file__).resolve().parents[2] / "reports/tables/msgarch"


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


def msgarch_scenarios(benchmark, msgarch_dir=MSGARCH_DIR):
    residuals = pd.read_parquet(msgarch_dir / "msgarch_residuals.parquet")
    if residuals.date.dt.tz is not None:
        residuals["date"] = residuals.date.dt.tz_convert("UTC").dt.tz_localize(None)
    pools = {fold: group.z.to_numpy() for fold, group in residuals.groupby("fold")}
    days = []
    for target, day in benchmark.groupby("target"):
        first = day.iloc[0]
        scenarios = first.scale * pools[first.fold]
        for _, row in day.iterrows():
            check_reconstruction(scenarios, row["var"], row.es, row.confidence)
        days.append((target, scenarios))
    return days
