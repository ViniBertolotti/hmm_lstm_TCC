"""Converte a saída do MS-GARCH (R) em previsões no formato do walk-forward.

Execute da raiz depois de r/msgarch_benchmark.R:
    .venv/Scripts/python scripts/msgarch_forecasts.py

VaR/ES principal por simulação histórica filtrada (FHS), como declarado no
projeto: cenários = vol MS-GARCH prevista na origem × resíduos padronizados do
desenvolvimento da mesma janela, com a mesma função de VaR/ES empírico dos
outros modelos. O VaR/ES paramétrico da densidade preditiva (Student-t por
estado) vai para um arquivo separado, como análise de sensibilidade.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.risk.var_es import empirical_var_es, losses_from_log_returns  # noqa: E402

DIR = ROOT / "reports/tables/msgarch"
SOURCE = ROOT / "reports/tables/walk_forward"
LEVELS = (0.95, 0.99)


def _naive_dates(frame, columns):
    """O arrow do R devolve as datas como instantes UTC com fuso local; volta a datas sem fuso."""
    for column in columns:
        if frame[column].dt.tz is not None:
            frame[column] = frame[column].dt.tz_convert("UTC").dt.tz_localize(None)
        if not frame[column].equals(frame[column].dt.normalize()):
            raise ValueError(f"{column}: datas deslocadas na volta do R.")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-asset", help="robustez ativo a ativo (padrão: carteira)")
    args = parser.parse_args()
    source, directory = SOURCE, DIR
    if args.target_asset:
        source = ROOT / "reports/tables/univariado" / args.target_asset
        directory = source / "msgarch"
    forecasts = _naive_dates(pd.read_parquet(directory / "msgarch_forecasts.parquet"), ["origin", "target"])
    residuals = _naive_dates(pd.read_parquet(directory / "msgarch_residuals.parquet"), ["date"])
    actual = pd.concat([pd.read_parquet(p) for p in sorted(source.glob("forecasts_fold*.parquet"))])
    actual = (actual[actual.model == "zero_mean_reference"]
              .drop_duplicates("target").set_index("target")[["actual_log_return", "actual_loss"]])

    rows, param_rows = [], []
    for fold, block in forecasts.groupby("fold"):
        z = residuals.loc[residuals.fold == fold, "z"].to_numpy()
        for f in block.itertuples():
            scenarios = losses_from_log_returns(f.vol_pct / 100 * z)
            common = {"origin": f.origin, "target": f.target, "fold": fold, "prediction": 0.0,
                      "dropout_sd": 0.0, "scale": f.vol_pct / 100, "regime": np.nan,
                      "calibration_n": len(z), "msgarch_states": f.K, "prob_high_state": f.prob_high_state}
            for level in LEVELS:
                risk = empirical_var_es(scenarios, level)
                rows.append(common | {"model": "ms_garch", "confidence": level, "residual_pool": "msgarch_fhs",
                                      "var": risk["var"], "es": risk["es"]})
                tag = f"{round(level * 100)}"
                param_rows.append(common | {"model": "ms_garch_param", "confidence": level,
                                            "residual_pool": "msgarch_student_t",
                                            "var": getattr(f, f"var{tag}_param"), "es": getattr(f, f"es{tag}_param")})
    for name, data in (("forecasts_msgarch.parquet", rows), ("forecasts_msgarch_param.parquet", param_rows)):
        frame = pd.DataFrame(data).join(actual, on="target")
        if frame[["actual_log_return", "var", "es"]].isna().any().any():
            raise ValueError(f"{name}: alvo sem realização ou risco ausente.")
        frame.to_parquet(directory / name, index=False)
    print(pd.DataFrame(rows).groupby("confidence")[["var", "es"]].describe().T.round(4))


if __name__ == "__main__":
    main()
