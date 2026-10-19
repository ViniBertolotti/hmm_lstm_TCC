"""Monta as séries de soja e café ajustadas por rolagem a partir da base pública do pysystemtrade.

Execute da raiz: .venv/Scripts/python scripts/build_public_futures.py

Os CSVs "multiple prices" trazem, a cada dia, o preço do contrato em uso e o do
contrato seguinte. Na data de rolagem, o fator novo/antigo é calculado com os
dois contratos cotados no mesmo dia (src/data/public_futures_chain.py), então o
salto da troca de contrato não vira retorno. Os arquivos ficam fixados num
commit e conferidos por SHA256; se o cache local divergir, o script para.

Limitações (registrar no Cap. 3): só há preço de fechamento (sem OHLCV), o
horário 23:00 da fonte é nocional e a base termina em 28/03/2024. O autor
deixou de atualizá-la citando dúvidas sobre o licenciamento dos dados
redistribuídos; o uso aqui é acadêmico, com citação da fonte.
"""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.collectors import adjust_futures_roll  # noqa: E402
from src.data.public_futures_chain import reconstruct_multiple, select_daily_snapshots  # noqa: E402

REPOSITORY = "robcarver17/pysystemtrade"
COMMIT = "a62269d27707806ba61f516791b71a9cd14cf268"
SOURCES = {
    "SOYBEAN": "911a09de70eca193a756bf88a12eca98eb195a248f3007ade5573c029f0b1bf4",
    "COFFEE": "3252a6b89fc32187ac1a427cd7ba99f7c65d64e3e8e5a03ca58aca9f6ff40264",
}
OUT = ROOT / "data" / "raw" / "futures_public"


def fetch(name: str, expected_sha256: str) -> Path:
    path = OUT / "source" / COMMIT / f"{name}.csv"
    if not path.exists():
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/data/futures/multiple_prices_csv/{name}.csv"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"{name}: SHA256 diferente do esperado ({digest}); arquivo não usado.")
    return path


def main():
    manifest = {"repository": REPOSITORY, "commit": COMMIT, "instruments": {}}
    for name, sha in SOURCES.items():
        source = fetch(name, sha)
        raw = pd.read_csv(source, parse_dates=["DATETIME"],
                          dtype={f"{role}_CONTRACT": str for role in ("PRICE", "FORWARD", "CARRY")})
        raw = raw.set_index("DATETIME").loc["2005-01-01":"2024-12-31"]
        chain, ledger = reconstruct_multiple(raw, adjust_futures_roll)
        daily = select_daily_snapshots(chain)
        daily[["price_index", "raw_price", "price_contract"]].to_parquet(OUT / f"{name}_daily.parquet")
        ledger.to_parquet(OUT / f"{name}_roll_ledger.parquet", index=False)
        valid = daily.price_index.dropna()
        manifest["instruments"][name] = {
            "source_sha256": sha, "first_date": str(valid.index[0].date()), "last_date": str(valid.index[-1].date()),
            "n_days": int(valid.size), "n_rolls": int(len(ledger)),
            "contract_months": sorted({c[4:6] for c in daily.price_contract.dropna().unique()}),
        }
        print(name, manifest["instruments"][name])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
