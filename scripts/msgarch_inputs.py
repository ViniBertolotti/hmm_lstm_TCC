"""Prepara as entradas do benchmark MS-GARCH (r/msgarch_benchmark.R).

Execute da raiz depois de run_walk_forward.py:
    .venv/Scripts/python scripts/msgarch_inputs.py

Usa as mesmas janelas e os mesmos alvos de teste dos outros modelos: o
desenvolvimento vai do início do treino ao fim da calibração; os alvos são os
pregões avaliados no walk-forward, para que a comparação seja pareada.

Robustez ativo a ativo:
    .venv/Scripts/python scripts/msgarch_inputs.py --target-asset KC=F
lê reports/tables/univariado/KC=F/ e escreve em reports/tables/univariado/KC=F/msgarch/.
"""
from pathlib import Path
import argparse
import json
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment import snapshot, write_json  # noqa: E402
SOURCE = ROOT / "reports/tables/walk_forward"
OUT = ROOT / "reports/tables/msgarch"


def paths(target_asset=None):
    """Pasta das previsões walk-forward e pasta de saída do MS-GARCH."""
    if not target_asset:
        return SOURCE, OUT
    source = ROOT / "reports/tables/univariado" / target_asset
    return source, source / "msgarch"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-asset")
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    source, out = paths(args.target_asset)
    source, out = args.input_dir or source, args.output_dir or out
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    benchmark_config = config['msgarch_benchmark'] | {
        'confidence_levels': config['risk']['confidence_levels'],
        'random_seed': config['random_seed']}
    if benchmark_config['distribution'] != 'student_t' or benchmark_config['method'] != 'FHS':
        raise ValueError('O benchmark implementado usa student_t com VaR/ES por FHS.')
    (out / 'inputs_config.json').write_text(json.dumps(benchmark_config, indent=2), encoding='utf-8')
    if args.target_asset:
        features = pd.read_parquet(ROOT / "data/processed/features.parquet")
        series = features[(args.target_asset, "return_1d")].rename("return_1d").to_frame()
    else:
        series = pd.read_parquet(ROOT / "data/processed/portfolio_return.parquet")[["return_1d"]]
    returns = series.rename_axis("date").reset_index()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    folds = pd.DataFrame([{"fold": m["fold"],
                           "dev_first": pd.Timestamp(m["block_dates"]["train"]["first"]),
                           "dev_last": pd.Timestamp(m["block_dates"]["calibration"]["last"]),
                           "test_first": pd.Timestamp(m["block_dates"]["test"]["first"]),
                           "test_last": pd.Timestamp(m["block_dates"]["test"]["last"])}
                          for m in manifest if m["model"] == "hmm_lstm"])
    frames = [pd.read_parquet(p) for p in sorted(source.glob("forecasts_fold*.parquet"))]
    reference = pd.concat(frames)
    reference = reference[(reference.model == "zero_mean_reference") & (reference.confidence == reference.confidence.min())]
    targets = reference[["fold", "origin", "target"]].sort_values(["fold", "target"])
    for name, frame in (("returns", returns), ("folds", folds), ("targets", targets)):
        frame.to_parquet(out / f"inputs_{name}.parquet", index=False)
    provenance = snapshot(ROOT, config,
        [out / f'inputs_{name}.parquet' for name in ('returns', 'folds', 'targets')] +
        [out / 'inputs_config.json', source / 'manifest.json'], options={'stage': 'msgarch_inputs'})
    write_json(out / 'inputs_manifest.json', provenance)
    print(f"{len(folds)} janelas, {len(targets)} alvos -> {out}")


if __name__ == "__main__":
    main()
