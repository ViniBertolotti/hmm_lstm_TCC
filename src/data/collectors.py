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


def adjust_futures_roll(prices: pd.Series, *, roll_factors: pd.Series | None = None) -> pd.Series:
    """Back-adjustment com fatores verificados (novo/antigo na MESMA data).

    O índice contém a primeira data do novo contrato. Sem fatores, preserva
    a série. Nunca infere rolagem a partir do retorno entre dois dias.
    Não fornece, por si só, uma série histórica point-in-time.
    """
    adjusted = prices.sort_index().copy()
    if adjusted.index.has_duplicates:
        raise ValueError("Datas duplicadas nos preços.")
    if roll_factors is None:
        return adjusted
    factors = roll_factors.sort_index(ascending=False)
    if factors.index.has_duplicates or not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError("Fatores de rolagem devem ser únicos, finitos e positivos.")
    for date, factor in factors.items():
        if date not in adjusted.index or date == adjusted.index[0] or pd.isna(adjusted.loc[date]):
            raise ValueError(f"Data de rolagem inválida: {date}")
        adjusted.loc[adjusted.index < date] *= factor
    return adjusted


def adjust_ohlc_roll(ohlc: pd.DataFrame, *, roll_factors: pd.Series | None = None) -> pd.DataFrame:
    """Aplica fatores documentados às quatro colunas de preço; preserva volume."""
    adjusted = ohlc.sort_index().copy()
    for col in ("Open", "High", "Low", "Close"):
        if col in adjusted:
            adjusted[col] = adjust_futures_roll(adjusted[col], roll_factors=roll_factors)
    return adjusted


def load_public_futures_close(path: str | os.PathLike) -> pd.DataFrame:
    """Índice de preço do futuro já ajustado por rolagem (só fechamento).

    Gerado por `scripts/build_public_futures.py` a partir da base pública do
    pysystemtrade. Volta como DataFrame com a coluna `Close` para entrar no
    mesmo fluxo de alinhamento dos demais ativos; Open/High/Low/Volume não
    existem nesta fonte.
    """
    daily = pd.read_parquet(path)
    return daily[["price_index"]].rename(columns={"price_index": "Close"}).dropna().sort_index()


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
