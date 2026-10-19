"""Reconstrução exploratória de preços múltiplos; não aprova a fonte para o TCC.

Cada troca usa cotações simultâneas do arquivo na observação imediatamente
anterior. A escala progressiva depende somente dos fatores já observados.
Isso não prova que o cronograma ex post era conhecido em tempo real, nem que
os timestamps representam ajustes oficiais ou horários de publicação.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLES = ("PRICE", "FORWARD", "CARRY")


def reconstruct_multiple(data: pd.DataFrame, adjust_futures_roll=None):
    """Retorna níveis por razão, fatores e reconstrução aditiva de conferência.

    Não preenche preços ausentes, não cria OHLCV, não estima spread a partir de
    retornos e não calcula P&L de uma carteira de contratos com margem/colateral.
    """
    required = [field for role in ROLES for field in (role, role + "_CONTRACT")]
    if not set(required).issubset(data.columns):
        raise ValueError("Colunas de preços e contratos incompletas.")
    if not isinstance(data.index, pd.DatetimeIndex) or len(data) < 2:
        raise ValueError("São necessários timestamps e pelo menos duas observações.")
    if data.index.hasnans or data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("Timestamps devem ser únicos, crescentes e sem NaT.")
    data = data.copy()
    for role in ROLES:
        values = pd.to_numeric(data[role], errors="raise").to_numpy(dtype=float)
        if np.isinf(values).any() or (values[np.isfinite(values)] <= 0).any():
            raise ValueError("Preços observados devem ser positivos e finitos.")
        data[role] = values
        contracts = data[role + "_CONTRACT"]
        if contracts.isna().any() or not contracts.map(lambda x: isinstance(x, str) and bool(x)).all():
            raise ValueError("Identificador de contrato ausente ou não textual.")
    if data.PRICE.notna().sum() == 0:
        raise ValueError("Não há preço ativo observado.")

    factor = pd.Series(1.0, index=data.index, name="factor")
    spread = pd.Series(0.0, index=data.index, name="spread")
    records = []
    positions = np.flatnonzero(data.PRICE_CONTRACT.ne(data.PRICE_CONTRACT.shift()).to_numpy())[1:]
    for i in positions:
        previous, current = data.iloc[i - 1], data.iloc[i]
        matches = [role for role in ROLES if previous[role + "_CONTRACT"] == current.PRICE_CONTRACT
                   and pd.notna(previous[role])]
        if not matches or pd.isna(previous.PRICE):
            raise ValueError("Faltam preços simultâneos imediatamente antes da troca.")
        new_prices = np.array([previous[role] for role in matches], dtype=float)
        if not np.allclose(new_prices, new_prices[0], rtol=0, atol=1e-10):
            raise ValueError("Cotações simultâneas contraditórias para o contrato novo.")
        old_price, new_price = float(previous.PRICE), float(new_prices[0])
        ratio = new_price / old_price
        factor.iloc[i] = ratio
        spread.iloc[i] = new_price - old_price
        records.append({"first_new_snapshot": data.index[i],
                        "factor_snapshot": data.index[i - 1],
                        "old_contract": previous.PRICE_CONTRACT,
                        "new_contract": current.PRICE_CONTRACT,
                        "old_price": old_price, "new_price_same_snapshot": new_price,
                        "factor": ratio, "spread": new_price - old_price,
                        "overlap_role": matches[0],
                        "price_missing_at_contract_change": bool(pd.isna(current.PRICE))})

    # A escala pode continuar após um preço ausente, sem inventar seu valor.
    # Retornos adjacentes a esse preço continuam ausentes.
    cumulative_log_factor = np.log(factor).cumsum()
    progressive = data.PRICE * np.exp(-cumulative_log_factor)
    progressive = (100 * progressive / progressive.dropna().iloc[0]).rename("price_index")
    # Nível retrospectivo é apenas uma apresentação da mesma cadeia de retornos.
    ratio_back = data.PRICE * np.exp(cumulative_log_factor.iloc[-1] - cumulative_log_factor)
    if adjust_futures_roll is not None:
        # O helper exige preço na data do fator. Para a conferência, usa somente
        # preços observados e transfere a escala acumulada à primeira observação
        # posterior à ausência. Nenhum valor é inserido na série original.
        observed = data.PRICE.dropna()
        changes = cumulative_log_factor.reindex(observed.index).diff()
        event_factors = np.exp(changes.loc[changes.notna() & changes.ne(0)])
        reference = adjust_futures_roll(observed, roll_factors=event_factors)
        np.testing.assert_allclose(ratio_back.reindex(observed.index), reference,
                                   rtol=1e-12, atol=1e-9, equal_nan=True)
    # Panama/diferença serve para conferir o arquivo ajustado distribuído pela fonte.
    additive = data.PRICE + spread.sum() - spread.cumsum()
    result = pd.DataFrame({"raw_price": data.PRICE, "price_contract": data.PRICE_CONTRACT,
                           "price_index": progressive, "ratio_back_adjusted": ratio_back,
                           "additive_reconstructed": additive, "factor": factor})
    result["log_return"] = np.log(progressive).diff()
    result["raw_log_return"] = np.log(data.PRICE).diff()
    result["is_roll"] = data.PRICE_CONTRACT.ne(data.PRICE_CONTRACT.shift())
    result.iloc[0, result.columns.get_loc("is_roll")] = False
    np.testing.assert_allclose(np.log(ratio_back).diff(), result.log_return,
                               rtol=1e-9, atol=1e-12, equal_nan=True)
    columns = ["first_new_snapshot", "factor_snapshot", "old_contract", "new_contract",
               "old_price", "new_price_same_snapshot", "factor", "spread", "overlap_role",
               "price_missing_at_contract_change"]
    ledger = pd.DataFrame(records, columns=columns)
    return result, ledger


def select_daily_snapshots(chain: pd.DataFrame):
    """Última linha disponível por data, inclusive se o preço está ausente.

    Não troca uma linha final ausente por outra anterior. O horário original é
    preservado: o resultado é diagnóstico, não uma série de settlements aprovada.
    """
    daily = chain.groupby(chain.index.normalize(), sort=True).tail(1).copy()
    daily["selected_timestamp"] = daily.index
    daily.index = daily.index.normalize()
    daily.index.name = "date"
    for price, ret in [("price_index", "log_return"), ("raw_price", "raw_log_return")]:
        daily[ret] = np.log(daily[price]).diff()
    daily["is_notional_2300"] = daily.selected_timestamp.dt.strftime("%H:%M:%S").eq("23:00:00")
    return daily
