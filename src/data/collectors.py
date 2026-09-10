"""Coleta de dados brutos (preços, câmbio, macro) para o pipeline HMM+LSTM.

Nenhum ticker/código de série é hard-coded fora deste módulo: os chamadores devem
passar as listas vindas de `config.yaml`. Cada `collect_*` faz apenas I/O de uma
fonte; `adjust_futures_roll` é uma função pura (sem I/O) para poder ser testada
sem rede.
"""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import requests
import truststore
import yfinance as yf
from bcb import sgs
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

# Em redes corporativas com proxy de inspeção HTTPS, o certificado apresentado é
# assinado por uma CA interna que não está no bundle do `certifi` (usado por
# padrão pelo `requests`), embora já esteja confiada no repositório de
# certificados do próprio Windows. `truststore` faz o `ssl`/`requests` usarem
# esse repositório nativo em vez de rejeitar a conexão.
truststore.inject_into_ssl()

# Códigos das séries no SGS do Banco Central (https://www3.bcb.gov.br/sgspub).
BCB_SGS_CODES = {
    "selic": 432,      # Meta Selic definida pelo Copom (% a.a.)
    "ipca": 433,        # IPCA - variação mensal (%)
    "ibc_br": 24363,    # IBC-Br - índice mensal dessazonalizado
}

# Séries do FRED (https://fred.stlouisfed.org). O DXY (ICE) não é publicado
# gratuitamente no FRED; DTWEXBGS (Broad Trade-Weighted U.S. Dollar Index) é o
# proxy padrão da literatura quando não se tem acesso a dados ICE pagos.
FRED_SERIES_IDS = {
    "us_treasury_10y": "DGS10",
    "dxy": "DTWEXBGS",
}

# CEPEA não tem API oficial: estes endpoints devolvem um .xls (BIFF/OLE2) com a
# série histórica completa do indicador, encontrados inspecionando manualmente
# https://www.cepea.org.br/br/indicador/{soja,cafe}.aspx -> "Série de preços".
# id=92 (soja) = indicador Paranaguá, paridade de exportação em R$ e US$/saca,
# mais comparável ao futuro internacional ZS=F do que o indicador Paraná (id=12).
# id=23 (café) = indicador Café Arábica (o robusta é id=24), compatível com KC=F.
CEPEA_SERIES_URLS = {
    "soja": "https://www.cepea.org.br/br/indicador/series/soja.aspx?id=92",
    "cafe_arabica": "https://www.cepea.org.br/br/indicador/series/cafe.aspx?id=23",
}


def collect_equities_fx(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Baixa OHLCV ajustado (Yahoo Finance) para cada equity/câmbio.

    Devolve um dict {ticker: DataFrame OHLCV}, não uma tabela só de Close: o
    Garman-Klass e o spread de Corwin-Schultz (Semana 3) precisam de
    Open/High/Low para todos os 7 ativos, não só para os futuros.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker")
    result = {}
    for ticker in tickers:
        df = raw[ticker] if isinstance(raw.columns, pd.MultiIndex) else raw
        result[ticker] = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
    return result


def collect_futures_raw(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Baixa OHLCV bruto (sem ajuste de rolagem) de um futuro no Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")


def _find_roll_jump_dates(close: pd.Series, jump_threshold: float) -> list:
    """Datas com log-retorno acima de `jump_threshold` (mais recentes primeiro)."""
    log_ret = np.log(close.dropna().sort_index()).diff()
    return sorted(log_ret[log_ret.abs() > jump_threshold].index, reverse=True)


def adjust_futures_roll(prices: pd.Series, jump_threshold: float = 0.08) -> pd.Series:
    """Aplica back-adjustment multiplicativo (por razão) a uma série contínua de futuro.

    O Yahoo Finance entrega o contrato contínuo sem ajustar as datas de rolagem,
    o que produz saltos de preço que não são retorno real e contaminam
    retornos/GARCH/HMM. Detecta esses saltos via log-retorno acima de
    `jump_threshold` e reescala multiplicativamente todo o histórico anterior a
    cada rolagem, preservando exatamente os retornos percentuais/log em toda a
    série e o nível do contrato mais recente.

    Um ajuste *aditivo* (deslocar por uma diferença fixa) foi tentado primeiro,
    mas em séries de ~20 anos com muitas rolagens (ZS=F, KC=F) o deslocamento
    acumulado empurra os preços do início da amostra para valores negativos —
    inviável para Garman-Klass/Corwin-Schultz, que precisam de log(High/Low) e
    log(Close/Open). O ajuste por razão nunca produz preços negativos (é uma
    multiplicação por um fator positivo) e é o método padrão da literatura para
    esse cenário.
    """
    adjusted = prices.dropna().sort_index().copy()
    for roll_date in _find_roll_jump_dates(adjusted, jump_threshold):
        pos = adjusted.index.get_loc(roll_date)
        if pos == 0:
            continue
        ratio = adjusted.iloc[pos] / adjusted.iloc[pos - 1]
        adjusted.iloc[:pos] = adjusted.iloc[:pos] * ratio
    return adjusted


def adjust_ohlc_roll(ohlc: pd.DataFrame, jump_threshold: float = 0.08) -> pd.DataFrame:
    """Aplica o mesmo ajuste de `adjust_futures_roll` a Open/High/Low/Close juntos.

    As rolagens são detectadas a partir do Close (mesma lógica de
    `adjust_futures_roll`); o mesmo fator multiplicativo é aplicado às quatro
    colunas de preço igualmente, para a barra OHLC continuar consistente
    (ex.: o Close ajustado não pode ficar fora do intervalo High/Low ajustados)
    e sempre positiva. Volume não é alterado.
    """
    adjusted = ohlc.dropna(subset=["Close"]).sort_index().copy()
    price_cols = [c for c in ("Open", "High", "Low", "Close") if c in adjusted.columns]
    for roll_date in _find_roll_jump_dates(adjusted["Close"], jump_threshold):
        pos = adjusted.index.get_loc(roll_date)
        if pos == 0:
            continue
        ratio = adjusted["Close"].iloc[pos] / adjusted["Close"].iloc[pos - 1]
        adjusted.iloc[:pos, adjusted.columns.get_indexer(price_cols)] *= ratio
    return adjusted


def collect_bcb_sgs(start: str, end: str) -> pd.DataFrame:
    """Baixa Selic, IPCA e IBC-Br do SGS/BCB (via pacote `python-bcb`).

    O SGS limita séries diárias (Selic) a janelas de no máximo 10 anos por
    requisição, mesmo quando pedidas junto com séries mensais — por isso o
    período é dividido em blocos de 9 anos (margem de segurança) e concatenado.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    chunks = []
    chunk_start = start_ts
    while chunk_start <= end_ts:
        chunk_end = min(chunk_start + pd.DateOffset(years=9), end_ts)
        chunks.append(sgs.get(BCB_SGS_CODES, start=chunk_start, end=chunk_end))
        chunk_start = chunk_end + pd.Timedelta(days=1)
    return pd.concat(chunks).sort_index()


def collect_implied_vol_us(start: str, end: str) -> pd.Series:
    """Baixa o VIX (CBOE) como proxy de volatilidade implícita americana."""
    df = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].rename("implied_vol_us").dropna()


def load_bloomberg_implied_vol_br(path: str | os.PathLike) -> pd.Series:
    """Lê a série de volatilidade implícita do Ibovespa exportada do Bloomberg.

    Fonte: terminal Bloomberg, ticker "IBOV Index", campo
    `30DAY_IMPVOL_100.0%MNY_DF` (vol. implícita ATM de 30 dias). Não há
    coletor automático — é um export manual (`.xlsx` com colunas
    `date`/`implied_vol_br`) que precisa ser atualizado periodicamente à mão
    e não deve ser versionado no git (ver `.gitignore`/`data/raw/`).
    """
    df = pd.read_excel(path)
    return df.set_index("date")["implied_vol_br"].sort_index()


def collect_fred(start: str, end: str) -> pd.DataFrame:
    """Baixa Treasury 10y e o proxy do DXY (DTWEXBGS) do FRED.

    Requer `FRED_API_KEY` definida em `.env` (ver `.env.example`).
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY não encontrada. Copie .env.example para .env e preencha a chave.")
    fred = Fred(api_key=api_key)
    series = {name: fred.get_series(series_id, start, end) for name, series_id in FRED_SERIES_IDS.items()}
    return pd.DataFrame(series)


def _parse_cepea_xls(content: bytes) -> pd.DataFrame:
    """Extrai (data, preço R$[, preço US$]) do .xls histórico devolvido pela CEPEA.

    O arquivo é um XLS binário (BIFF/OLE2) genuíno, mas gerado por uma ferramenta
    que o xlrd classifica como "corrompido" por não seguir à risca o layout do
    compound document — por isso `ignore_workbook_corruption=True` (o conteúdo em
    si é lido corretamente; células numéricas já vêm como float, sem formatação
    BR de milhar/decimal para desfazer).
    """
    raw = pd.read_excel(
        io.BytesIO(content),
        header=None,
        engine="xlrd",
        engine_kwargs={"ignore_workbook_corruption": True},
    )
    header_row = raw.index[raw.iloc[:, 0].astype(str).str.strip().str.lower() == "data"]
    if len(header_row) == 0:
        raise ValueError("Não encontrei a linha de cabeçalho ('Data') no .xls da CEPEA.")
    start_row = header_row[0] + 1

    data = raw.iloc[start_row:, :3].copy()
    data.columns = ["date", "price_brl", "price_usd"][: data.shape[1]]
    data["date"] = pd.to_datetime(data["date"], dayfirst=True, errors="coerce")
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.dropna(subset=["date"]).set_index("date").sort_index()


# A CEPEA devolve 403 para requisições sem um User-Agent de navegador (bot protection).
_CEPEA_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def collect_cepea_spot(commodity: str) -> pd.DataFrame:
    """Baixa a série histórica spot (BRL) da CEPEA para 'soja' ou 'cafe_arabica'.

    A CEPEA não expõe filtro por data: o endpoint sempre devolve a série
    completa, que é filtrada depois no notebook de orquestração.
    """
    url = CEPEA_SERIES_URLS[commodity]
    response = requests.get(url, headers=_CEPEA_HEADERS, timeout=30)
    response.raise_for_status()
    return _parse_cepea_xls(response.content)
