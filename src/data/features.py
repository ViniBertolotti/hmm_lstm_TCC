"""Engenharia de features (Semana 3, Passada 1) — retornos, volatilidade e liquidez.

Cada função opera sobre um único ativo (uma série ou um DataFrame OHLCV) e é
pura (sem I/O): a orquestração multi-ativo/multi-fonte fica no notebook
`02_feature_engineering.ipynb`. `build_asset_features` combina todas as
funções abaixo na ordem descrita em `PLANO.md` > Semana 3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model


def align_and_fill_ohlc(ohlc: pd.DataFrame, start: str, end: str, max_gap_days: int = 3) -> pd.DataFrame:
    """Alinha um OHLC ao calendário de dias úteis [start, end] e trata missing.

    Regra da Semana 3 (`PLANO.md`): buracos de até `max_gap_days` dias úteis
    são preenchidos por interpolação linear; buracos maiores ficam `NaN`
    (mesma política de "NaN estrutural, trim só no dataset final" já usada
    para o burn-in do GARCH e para o `implied_vol_br` pré-2015).
    """
    business_days = pd.bdate_range(start, end)
    aligned = ohlc.reindex(business_days)
    return aligned.interpolate(method="linear", limit=max_gap_days, limit_direction="forward", limit_area="inside")


def compute_return_lags(prices: pd.Series, lags: tuple[int, ...] = (1, 5, 21)) -> pd.DataFrame:
    """Retornos log em múltiplos horizontes: `return_{k}d` = ln(P_t / P_{t-k}).

    `lags=(1,5,21)` produz retorno diário, semanal (~5 pregões) e mensal
    (~21 pregões), todos conhecidos em `t` (sem lookahead).
    """
    log_prices = np.log(prices)
    return pd.DataFrame({f"return_{k}d": log_prices.diff(k) for k in lags})


def rolling_realized_volatility(
    return_1d: pd.Series, windows: tuple[int, ...] = (21, 63), annualize: bool = True
) -> pd.DataFrame:
    """Volatilidade realizada: desvio-padrão móvel do retorno diário, anualizado (√252)."""
    factor = np.sqrt(252) if annualize else 1.0
    return pd.DataFrame({f"realized_vol_{w}d": return_1d.rolling(w).std() * factor for w in windows})


def garman_klass_volatility(
    ohlc: pd.DataFrame, windows: tuple[int, ...] = (21, 63), annualize: bool = True
) -> pd.DataFrame:
    """Estimador de volatilidade de Garman & Klass (1980), a partir de OHLC.

    Usa a variância diária GK_t = 0.5·ln(H/L)² − (2·ln2−1)·ln(C/O)², suavizada
    em janela móvel antes da raiz quadrada (mais estável que aplicar a raiz
    diariamente).
    """
    log_hl = np.log(ohlc["High"] / ohlc["Low"])
    log_co = np.log(ohlc["Close"] / ohlc["Open"])
    daily_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    factor = 252 if annualize else 1.0
    return pd.DataFrame(
        {f"garman_klass_vol_{w}d": np.sqrt((daily_var.rolling(w).mean() * factor).clip(lower=0)) for w in windows}
    )


def corwin_schultz_spread(ohlc: pd.DataFrame) -> pd.Series:
    """Spread de bid-ask implícito de Corwin & Schultz (2012), via High/Low de 2 dias.

    Proxy de liquidez: o Yahoo Finance não fornece bid/ask histórico diário,
    só OHLCV — este estimador é o método acadêmico padrão para essa limitação.
    Valores negativos (artefato do estimador em dias de baixa volatilidade)
    são zerados, conforme a prática usual na literatura.
    """
    log_hl = np.log(ohlc["High"] / ohlc["Low"])
    beta = log_hl**2 + log_hl.shift(1) ** 2
    high_2d = ohlc["High"].rolling(2).max()
    low_2d = ohlc["Low"].rolling(2).min()
    gamma = np.log(high_2d / low_2d) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0).rename("corwin_schultz_spread")


def rolling_skew_kurtosis(return_1d: pd.Series, windows: tuple[int, ...] = (21, 63)) -> pd.DataFrame:
    """Assimetria e curtose móveis do retorno diário."""
    out = {}
    for w in windows:
        out[f"skew_{w}d"] = return_1d.rolling(w).skew()
        out[f"kurtosis_{w}d"] = return_1d.rolling(w).kurt()
    return pd.DataFrame(out)


def garch_conditional_vol_walkforward(
    return_1d: pd.Series, refit_every: int = 21, burn_in: int = 500
) -> pd.Series:
    """Volatilidade condicional de um GARCH(1,1) em janela expansiva, sem lookahead.

    Reajusta o modelo a cada `refit_every` pregões (padrão: mensal) usando só
    dados até aquele ponto, e usa o forecast de 1 passo à frente para os dias
    até o próximo refit — muito mais rápido que reajustar diariamente, e ainda
    assim sem vazar informação futura para o passado (cada refit só enxerga o
    que já aconteceu). As primeiras `burn_in` observações ficam `NaN`
    (precisam de histórico mínimo para o primeiro fit).
    """
    returns = return_1d.dropna()
    returns_pct = returns * 100  # arch recomenda reescalar log-retornos pequenos para evitar problemas numéricos
    cond_vol = pd.Series(index=returns.index, dtype=float)

    for start_idx in range(burn_in, len(returns), refit_every):
        train = returns_pct.iloc[:start_idx]
        model = arch_model(train, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        result = model.fit(disp="off")

        horizon = min(refit_every, len(returns) - start_idx)
        forecast = result.forecast(horizon=horizon, reindex=False)
        forecast_vol_pct = np.sqrt(forecast.variance.values[-1])
        cond_vol.iloc[start_idx : start_idx + horizon] = forecast_vol_pct / 100

    return cond_vol.rename("garch_cond_vol")


def build_asset_features(
    ohlc: pd.DataFrame,
    lags: tuple[int, ...] = (1, 5, 21),
    windows: tuple[int, ...] = (21, 63),
    garch_refit_every: int = 21,
    garch_burn_in: int = 500,
) -> pd.DataFrame:
    """Monta todas as features da Passada 1 para um único ativo, a partir do OHLCV bruto."""
    returns = compute_return_lags(ohlc["Close"], lags=lags)
    return_1d = returns["return_1d"]

    parts = [
        returns,
        rolling_realized_volatility(return_1d, windows),
        garman_klass_volatility(ohlc, windows),
        garch_conditional_vol_walkforward(return_1d, garch_refit_every, garch_burn_in).to_frame(),
        rolling_skew_kurtosis(return_1d, windows),
        corwin_schultz_spread(ohlc).to_frame(),
        (ohlc["Volume"] / ohlc["Volume"].rolling(63).mean()).rename("volume_normalized").to_frame(),
    ]
    return pd.concat(parts, axis=1)
