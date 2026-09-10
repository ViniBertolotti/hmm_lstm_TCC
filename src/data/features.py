"""Engenharia de features (Semana 3, Passada 1) — retornos, volatilidade e liquidez.

Cada função opera sobre um único ativo (uma série ou um DataFrame OHLCV) e é
pura (sem I/O): a orquestração multi-ativo/multi-fonte fica no notebook
`02_feature_engineering.ipynb`. `build_asset_features` combina todas as
funções abaixo na ordem descrita em `PLANO.md` > Semana 3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from arch import arch_model

_PRICE_COLS = ("Open", "High", "Low", "Close")


def get_b3_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Dias de pregão reais da B3 (não um calendário genérico seg-sex).

    A carteira é definida na perspectiva de um investidor brasileiro: o
    painel só deve existir em dias em que ele conseguiria de fato negociar.
    """
    schedule = mcal.get_calendar("B3").schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(schedule.index.tz_localize(None))


def align_and_fill_ohlc(ohlc: pd.DataFrame, start: str, end: str, max_gap_days: int = 3) -> pd.DataFrame:
    """Alinha um OHLC ao calendário real de pregão da B3 e trata missing.

    Só usa informação passada: preenche buracos de até `max_gap_days` dias de
    pregão por *forward-fill* (carrega o último preço observado — o retorno
    do ativo nesse dia fica implicitamente 0%, uma aproximação honesta para
    "não sabemos o que aconteceu"). Nunca interpola linearmente nem usa
    *backward-fill* — isso usaria um preço futuro para preencher o passado
    (lookahead bias). Buracos maiores que `max_gap_days` ficam `NaN` (mesma
    política de "NaN estrutural, trim só no dataset final de modelagem" já
    usada para o burn-in do GARCH e para o `implied_vol_br` pré-2015).

    Um dia em que a bolsa de origem do ativo (CBOT/ICE, para os futuros)
    esteve fechada mas a B3 esteve aberta é tratado da mesma forma — mercado
    fechado para aquele ativo específico, não um erro de dado.
    """
    trading_days = get_b3_trading_days(start, end)
    aligned = ohlc.reindex(trading_days)

    price_cols = [c for c in _PRICE_COLS if c in aligned.columns]
    is_missing = aligned[price_cols].isna().all(axis=1)

    # ffill(limit=N) preencheria só os N primeiros dias de QUALQUER buraco, mesmo um longo.
    # Aqui o critério é "tudo ou nada" por buraco: agrupa runs consecutivos de dias faltantes
    # e só preenche os runs com no máximo `max_gap_days` dias no total.
    run_id = (is_missing != is_missing.shift()).cumsum()
    run_length = is_missing.groupby(run_id).transform("sum")
    fillable = is_missing & (run_length <= max_gap_days)

    ffilled = aligned[price_cols].ffill()
    aligned.loc[fillable, price_cols] = ffilled.loc[fillable]

    if "Volume" in aligned.columns:
        aligned.loc[fillable, "Volume"] = aligned.loc[fillable, "Volume"].fillna(0.0)

    return aligned


def convert_ohlc_to_brl(ohlc_usd: pd.DataFrame, fx_close_brl_per_usd: pd.Series) -> pd.DataFrame:
    """Converte um OHLC em USD para BRL-equivalente usando o câmbio do próprio dia.

    Preco_BRL(t) = Preco_USD(t) × FX(t). Como é o mesmo FX aplicado a
    Open/High/Low/Close do mesmo dia, razões intradiárias (Garman-Klass,
    Corwin-Schultz) não mudam — mas retornos, GARCH e skew/curtose passam a
    incorporar a volatilidade cambial, essencial para medir risco na
    perspectiva de um investidor brasileiro (ver `PLANO.md` > Pendências).
    """
    converted = ohlc_usd.copy()
    fx_aligned = fx_close_brl_per_usd.reindex(ohlc_usd.index)
    price_cols = [c for c in _PRICE_COLS if c in converted.columns]
    converted[price_cols] = converted[price_cols].mul(fx_aligned, axis=0)
    return converted


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
    """Volatilidade condicional de um GARCH(1,1), parâmetros reajustados
    periodicamente e variância condicional atualizada diariamente, sem lookahead.

    A cada `refit_every` pregões (padrão: mensal), reestima (ω, α, β) por
    máxima verossimilhança usando só dados até aquele ponto. Entre um refit e
    o próximo, a variância condicional segue a recursão
    `h_t = ω + α·r_{t-1}² + β·h_{t-1}` **dia a dia, com os retornos
    realmente observados** — mais fiel à dinâmica do processo do que um
    forecast multi-passos-à-frente (que diverge do caminho real quando há
    choques entre um refit e o outro), e ainda assim causal: h_t só depende
    de r's e h's estritamente anteriores. As primeiras `burn_in` observações
    ficam `NaN` (precisam de histórico mínimo para o primeiro fit).
    """
    returns = return_1d.dropna()
    returns_pct = returns * 100  # arch recomenda reescalar log-retornos pequenos para evitar problemas numéricos
    cond_vol = pd.Series(index=returns.index, dtype=float)

    for start_idx in range(burn_in, len(returns), refit_every):
        train = returns_pct.iloc[:start_idx]
        model = arch_model(train, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        result = model.fit(disp="off")

        omega = result.params["omega"]
        alpha1 = result.params["alpha[1]"]
        beta1 = result.params["beta[1]"]

        h_prev = result.conditional_volatility.iloc[-1] ** 2  # h_{start_idx-1}, do fit in-sample
        r_prev = train.iloc[-1]  # r_{start_idx-1}, observado

        horizon = min(refit_every, len(returns) - start_idx)
        h_path = np.empty(horizon)
        for step in range(horizon):
            h_t = omega + alpha1 * r_prev**2 + beta1 * h_prev
            h_path[step] = h_t
            r_prev = returns_pct.iloc[start_idx + step]  # retorno recém-realizado nesse dia
            h_prev = h_t

        cond_vol.iloc[start_idx : start_idx + horizon] = np.sqrt(h_path) / 100

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
