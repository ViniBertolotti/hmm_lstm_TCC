"""Prepara as entradas do benchmark MS-GARCH (r/msgarch_benchmark.R).

Execute da raiz depois de run_walk_forward.py:
    .venv/Scripts/python scripts/msgarch_inputs.py

Usa as mesmas janelas e os mesmos alvos de teste dos outros modelos: o
desenvolvimento vai do início do treino ao fim da calibração; os alvos são os
pregões avaliados no walk-forward, para que a comparação seja pareada.
"""
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/tables/walk_forward"
OUT = ROOT / "reports/tables/msgarch"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    portfolio = pd.read_parquet(ROOT / "data/processed/portfolio_return.parquet")
    returns = portfolio[["return_1d"]].rename_axis("date").reset_index()
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    folds = pd.DataFrame([{"fold": m["fold"],
                           "dev_first": pd.Timestamp(m["block_dates"]["train"]["first"]),
                           "dev_last": pd.Timestamp(m["block_dates"]["calibration"]["last"]),
                           "test_first": pd.Timestamp(m["block_dates"]["test"]["first"]),
                           "test_last": pd.Timestamp(m["block_dates"]["test"]["last"])}
                          for m in manifest if m["model"] == "hmm_lstm"])
    frames = [pd.read_parquet(p) for p in sorted(SOURCE.glob("forecasts_fold*.parquet"))]
    reference = pd.concat(frames)
    reference = reference[(reference.model == "zero_mean_reference") & (reference.confidence == reference.confidence.min())]
    targets = reference[["fold", "origin", "target"]].sort_values(["fold", "target"])
    for name, frame in (("returns", returns), ("folds", folds), ("targets", targets)):
        frame.to_parquet(OUT / f"inputs_{name}.parquet", index=False)
    print(f"{len(folds)} janelas, {len(targets)} alvos -> {OUT}")


if __name__ == "__main__":
    main()
