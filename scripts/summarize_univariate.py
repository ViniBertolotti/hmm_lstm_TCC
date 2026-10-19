"""Resume a robustez ativo a ativo.

Execute da raiz, depois de, para cada ativo:
    run_walk_forward.py --target-asset A --output-dir reports/tables/univariado/A
    msgarch_inputs.py --target-asset A; Rscript r/msgarch_benchmark.R reports/tables/univariado/A/msgarch
    msgarch_forecasts.py --target-asset A
    evaluate_walk_forward.py --input-dir reports/tables/univariado/A --msgarch-dir reports/tables/univariado/A/msgarch

Junta as avaliações em reports/tables/univariado/resumo_cobertura.csv e resumo_dm.csv.
O regime de estresse é o da carteira (o mesmo HMM do experimento principal).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "reports/tables/univariado"
ASSETS = ("SLCE3.SA", "JBSS32.SA", "MBRF3.SA", "SMTO3.SA", "ZS=F", "KC=F", "BRL=X")


def main():
    coverage, dm = [], []
    for asset in ASSETS:
        evaluation = DIR / asset / "evaluation_v3"
        if not (evaluation / "backtest_cobertura.csv").exists():
            print(f"{asset}: avaliação ausente, ignorado")
            continue
        coverage.append(pd.read_csv(evaluation / "backtest_cobertura.csv").assign(ativo=asset))
        dm.append(pd.read_csv(evaluation / "backtest_comparacao_dm.csv").assign(ativo=asset))
    coverage, dm = pd.concat(coverage, ignore_index=True), pd.concat(dm, ignore_index=True)
    keep = ["ativo", "nivel", "segmento", "modelo", "n", "violacoes", "taxa", "kupiec_p",
            "independencia_p", "cobertura_cond_p", "acerbi_szekely_z2", "es_p"]
    coverage[keep].to_csv(DIR / "resumo_cobertura.csv", index=False)
    dm[["ativo", "nivel", "segmento", "escore", "a", "b", "diferenca_a_menos_b", "dm_p"]].to_csv(
        DIR / "resumo_dm.csv", index=False)
    wide = coverage[coverage.segmento == "todos"].pivot_table(index="ativo", columns=["nivel", "modelo"], values="taxa")
    print((wide * 100).round(1).to_string())


if __name__ == "__main__":
    main()
