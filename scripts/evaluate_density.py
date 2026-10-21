"""PIT e CRPS diários dos quatro modelos (distribuição prevista inteira).

Execute da raiz depois de evaluate_walk_forward.py:
    .venv/Scripts/python scripts/evaluate_density.py
    .venv/Scripts/python scripts/evaluate_density.py --input-dir reports/tables/univariado/KC=F \
        --msgarch-dir reports/tables/univariado/KC=F/msgarch

Saída em <input-dir>/density/: pit_crps.parquet (um valor por modelo e dia), resumo.csv
(uniformidade do PIT por KS e Berkowitz, CRPS médio) e crps_dm.csv (Diebold–Mariano no
CRPS contra o MS-GARCH e entre as redes). O regime de estresse é o mesmo da avaliação.
"""
import argparse
import json
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.features import get_b3_trading_days  # noqa: E402
from src.risk.backtests import diebold_mariano  # noqa: E402
from src.risk.density import berkowitz_test, crps_ensemble, ks_uniform, pit_ensemble  # noqa: E402
from src.risk.scenarios import MSGARCH_DIR, fhs_scenarios, msgarch_scenarios  # noqa: E402

PAIRS = (("hmm_lstm", "ms_garch"), ("lstm", "ms_garch"), ("zero_mean_reference", "ms_garch"),
         ("hmm_lstm", "lstm"), ("hmm_lstm", "zero_mean_reference"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "reports/tables/walk_forward")
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument("--msgarch-dir", type=Path, default=MSGARCH_DIR)
    parser.add_argument("--min-regime-pool", type=int, default=100)
    parser.add_argument("--dm-max-lag", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, msgarch_dir = args.input_dir.resolve(), args.msgarch_dir.resolve()

    frame = pd.concat([pd.read_parquet(p) for p in sorted(source.glob("forecasts_fold*.parquet"))], ignore_index=True)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    calibrations = pd.concat([pd.read_parquet(p) for p in sorted(source.glob("calibration_fold*.parquet"))],
                             ignore_index=True)
    benchmark = pd.read_parquet(msgarch_dir / "forecasts_msgarch.parquet")
    sessions = get_b3_trading_days(calibrations.origin.min(), frame.target.max())
    days = fhs_scenarios(frame, calibrations, manifest, sessions=sessions, min_regime_pool=args.min_regime_pool)
    days["ms_garch"] = msgarch_scenarios(benchmark, msgarch_dir)

    n_states = {m["fold"]: m["n_states"] for m in manifest if m["model"] == "hmm_lstm"}
    hmm = frame[(frame.model == "hmm_lstm") & (frame.confidence == frame.confidence.min())].set_index("target")
    realized = hmm.actual_log_return
    stress = (hmm.regime == hmm.fold.map(n_states) - 1)

    rng = np.random.default_rng(args.seed)
    rows = []
    for model, model_days in days.items():
        for target, scenarios in model_days:
            y = float(realized.loc[target])
            rows.append({"model": model, "target": target, "pit": pit_ensemble(scenarios, y, rng),
                         "crps": crps_ensemble(scenarios, y), "estresse": bool(stress.loc[target])})
    daily = pd.DataFrame(rows).sort_values(["model", "target"]).reset_index(drop=True)
    if daily.groupby("model").size().nunique() != 1:
        raise ValueError("Modelos com números de dias diferentes.")

    segments = {"todos": daily.target.unique(), "estresse": stress.index[stress], "demais_regimes": stress.index[~stress]}
    summary, comparisons = [], []
    calendar = get_b3_trading_days(daily.target.min() - pd.Timedelta(days=10), daily.target.max())
    for segment, dates in segments.items():
        sub = daily[daily.target.isin(dates)]
        for model, d in sub.groupby("model"):
            ks = ks_uniform(d.pit)
            bk = berkowitz_test(d.sort_values('target').set_index('target').pit, sessions=calendar)
            summary.append({"segmento": segment, "modelo": model, "n": len(d), "crps_medio": d.crps.mean(),
                            "ks": ks["statistic"], "ks_p": ks["p_value"], "berkowitz_lr": bk["lr"],
                            "berkowitz_p": bk["p_value"], "berkowitz_mu": bk["mu"], "berkowitz_sigma": bk["sigma"],
                            "berkowitz_rho": bk["rho"], 'berkowitz_n_pairs': bk['n'],
                            'berkowitz_gap_pairs_excluded': bk['n_gap_pairs_excluded'],
                            "pit_abaixo_5%": (d.pit < .05).mean(),
                            "pit_acima_95%": (d.pit > .95).mean()})
        crps = sub.pivot(index="target", columns="model", values="crps")
        for a, b in PAIRS:
            dm = diebold_mariano(crps[a], crps[b], sessions=calendar, max_lag=args.dm_max_lag)
            comparisons.append({"segmento": segment, "a": a, "b": b, "crps_a": crps[a].mean(), "crps_b": crps[b].mean(),
                                "diferenca_a_menos_b": dm["mean_loss_a_minus_b"], "dm": dm["statistic"],
                                "dm_p": dm["p_value_two_sided"], "leitura": "diferença positiva favorece b"})

    out = args.output_dir or source / 'density'
    out.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out / "pit_crps.parquet", index=False)
    pd.DataFrame(summary).to_csv(out / "resumo.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "crps_dm.csv", index=False)
    inputs = [source / 'manifest.json', *source.glob('forecasts_fold*.parquet'),
              *source.glob('calibration_fold*.parquet'),
              msgarch_dir / 'forecasts_msgarch.parquet', msgarch_dir / 'msgarch_residuals.parquet']
    metadata = {'berkowitz_calendar': 'adjacent_sessions_only',
                'seed': args.seed, 'dm_max_lag': args.dm_max_lag,
                'inputs_sha256': {str(p): sha256(p.read_bytes()).hexdigest() for p in inputs},
                'sources_sha256': {str(p): sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__), ROOT / 'src/risk/density.py', ROOT / 'src/risk/scenarios.py',
                     ROOT / 'src/risk/backtests.py', ROOT / 'src/risk/calibration.py', ROOT / 'src/risk/var_es.py']}}
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    pd.set_option("display.width", 220)
    print(pd.DataFrame(summary).query("segmento == 'todos'").round(4).to_string(index=False))
    print(pd.DataFrame(comparisons).query("segmento == 'todos'").round(6).to_string(index=False))


if __name__ == "__main__":
    main()
