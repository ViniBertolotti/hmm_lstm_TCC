"""Backtests do experimento walk-forward (H1-H3).

Execute da raiz depois de run_walk_forward.py:
    .venv/Scripts/python scripts/evaluate_walk_forward.py

H1: cobertura do VaR a 95%/99% (Kupiec) e cobertura condicional (Christoffersen).
H2: o mesmo dentro do regime de maior volatilidade (estado mais alto do HMM em cada janela).
H3: comparação nos regimes de menor volatilidade, via Diebold-Mariano sobre o escore
    quantílico. Não rejeitar a igualdade não prova equivalência; o relatório guarda o
    intervalo da diferença para essa leitura.

Os regimes vêm do filtro Forward da HMM-LSTM na origem de cada previsão e são usados
também para separar os dias dos outros modelos, para que todos sejam avaliados nos
mesmos pregões.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.features import get_b3_trading_days  # noqa: E402
from src.risk.backtests import acerbi_szekely_unconditional, diebold_mariano, quantile_score, var_coverage  # noqa: E402

DIR = ROOT / "reports/tables/walk_forward"
MODELS = ("hmm_lstm", "lstm", "zero_mean_reference")
PAIRS = (("hmm_lstm", "lstm"), ("hmm_lstm", "zero_mean_reference"), ("lstm", "zero_mean_reference"))


def coverage_row(subset, sessions, level):
    if subset.empty:
        return {"n": 0}
    losses = subset.set_index("target")["actual_loss"]
    var = subset.set_index("target")["var"]
    cov = var_coverage(losses, var, level, sessions=sessions)
    es = acerbi_szekely_unconditional(losses.to_numpy(), var.to_numpy(), subset["es"].to_numpy(), level) \
        if (subset["es"] > 0).all() else {}
    return {"n": cov["n_observations"], "violacoes": cov["n_failures"], "esperado": round(cov["expected_failures_under_nominal_coverage"], 1),
            "taxa": cov["failure_rate"], "kupiec_p": cov["kupiec"]["p_value_chi2_1"],
            "independencia_p": cov.get("independence", {}).get("p_value_chi2_1"),
            "cobertura_cond_p": cov.get("conditional_coverage", {}).get("p_value_chi2_2"),
            "acerbi_szekely_z2": es.get("statistic_z_unconditional")}


def dm_row(frame, a, b, level, sessions, dates):
    if len(dates) < 10:
        return {"status": "amostra_insuficiente"}
    scores = {}
    for model in (a, b):
        s = frame[(frame.model == model) & (frame.confidence == level)].set_index("target").loc[dates]
        scores[model] = pd.Series(quantile_score(s.actual_loss.to_numpy(), s["var"].to_numpy(), level), index=s.index)
    if len(dates) < 10:
        return {"status": "amostra_insuficiente"}
    dm = diebold_mariano(scores[a], scores[b], sessions=sessions, max_lag=5)
    diff = (scores[a] - scores[b]).mean()
    return {"media_escore_a": float(scores[a].mean()), "media_escore_b": float(scores[b].mean()),
            "diferenca_a_menos_b": float(diff), "dm_estatistica": dm.get("statistic"), "dm_p": dm.get("p_value_two_sided"),
            "leitura": "diferença positiva favorece b"}


def main():
    frame = pd.concat([pd.read_parquet(p) for p in sorted(DIR.glob("forecasts_fold*.parquet"))], ignore_index=True)
    manifest = json.loads((DIR / "manifest.json").read_text(encoding="utf-8"))
    n_states = {m["fold"]: m["n_states"] for m in manifest if m["model"] == "hmm_lstm"}
    dup = frame.duplicated(["model", "confidence", "target"]).sum()
    if dup:
        raise ValueError(f"{dup} previsões duplicadas para o mesmo alvo — janelas de teste se sobrepõem.")
    sessions = get_b3_trading_days(frame.target.min() - pd.Timedelta(days=10), frame.target.max())

    regime = (frame[(frame.model == "hmm_lstm") & (frame.confidence == frame.confidence.min())]
              .assign(top=lambda d: d.fold.map(n_states) - 1)
              .assign(estresse=lambda d: d.regime == d.top)
              .set_index("target")[["regime", "estresse", "fold"]])
    frame = frame.join(regime[["estresse"]], on="target")

    rows, dm_rows = [], []
    segments = {"todos": regime.index, "estresse": regime.index[regime.estresse], "demais_regimes": regime.index[~regime.estresse]}
    for level in sorted(frame.confidence.unique()):
        for segment, dates in segments.items():
            for model in MODELS:
                subset = frame[(frame.model == model) & (frame.confidence == level) & frame.target.isin(dates)]
                rows.append({"nivel": level, "segmento": segment, "modelo": model} | coverage_row(subset, sessions, level))
            for a, b in PAIRS:
                dm_rows.append({"nivel": level, "segmento": segment, "a": a, "b": b} | dm_row(frame, a, b, level, sessions, dates))

    coverage = pd.DataFrame(rows)
    comparisons = pd.DataFrame(dm_rows)
    coverage.to_csv(DIR / "backtest_cobertura.csv", index=False)
    comparisons.to_csv(DIR / "backtest_comparacao_dm.csv", index=False)
    pools = frame[frame.model == "hmm_lstm"].residual_pool.value_counts(normalize=True).round(3).to_dict()
    summary = {"periodo_teste": [str(frame.target.min().date()), str(frame.target.max().date())],
               "n_janelas": int(frame.fold.nunique()), "dias_de_teste": int(regime.shape[0]),
               "dias_em_estresse": int(regime.estresse.sum()), "uso_da_cauda_por_regime": pools}
    (DIR / "backtest_resumo.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(json.dumps(summary, ensure_ascii=False))
    print(coverage.round(4).to_string(index=False))
    print(comparisons[["nivel", "segmento", "a", "b", "diferenca_a_menos_b", "dm_p"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
