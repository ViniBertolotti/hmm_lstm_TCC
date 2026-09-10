"""Validação de cobertura temporal e detecção de gaps em séries de preços/macro."""
from __future__ import annotations

import pandas as pd


def check_coverage(
    series: pd.Series,
    start: str,
    end: str,
    name: str,
    max_gap_periods: int = 10,
    tolerance_days: int = 7,
    freq: str = "B",
) -> dict:
    """Resume cobertura temporal de `series` no período [start, end] e sinaliza gaps.

    Um gap é "crítico" quando há mais de `max_gap_periods` períodos consecutivos
    (na frequência `freq`) sem observação dentro do intervalo coberto pela série.
    Use `freq="B"` (padrão) para séries diárias/dias úteis e `freq="MS"` para
    séries mensais (BCB IPCA/IBC-Br) — aplicar a checagem diária a uma série
    mensal produziria "gaps" em todo mês, que não são um problema real de dados.
    """
    series = series.dropna()
    if series.empty:
        return {
            "name": name,
            "n_obs": 0,
            "coverage_start": None,
            "coverage_end": None,
            "covers_requested_range": False,
            "n_missing_periods": None,
            "critical_gaps": [],
        }

    idx = pd.DatetimeIndex(series.index).sort_values()
    actual_start, actual_end = idx.min(), idx.max()
    expected_start, expected_end = pd.Timestamp(start), pd.Timestamp(end)
    tolerance = pd.Timedelta(days=tolerance_days)

    expected_periods = pd.date_range(max(actual_start, expected_start), min(actual_end, expected_end), freq=freq)
    if freq == "MS":
        # normaliza a série para o 1º dia do mês antes de comparar com a grade mensal
        idx = idx.to_period("M").to_timestamp()
    missing = expected_periods.difference(idx)

    critical_gaps = []
    if len(missing) > 0:
        # Agrupa por posição consecutiva dentro de `expected_periods`, não por diferença de
        # dias corridos: p.ex. uma sexta seguida de uma segunda são períodos "B" consecutivos,
        # mas têm 3 dias corridos de diferença, o que quebraria o bloco incorretamente.
        missing_positions = pd.Series(expected_periods.get_indexer(missing), index=missing)
        block_id = (missing_positions.diff() != 1).cumsum()
        for _, block in missing_positions.groupby(block_id):
            if len(block) > max_gap_periods:
                critical_gaps.append((block.index[0].date(), block.index[-1].date(), len(block)))

    return {
        "name": name,
        "n_obs": len(series),
        "coverage_start": actual_start.date(),
        "coverage_end": actual_end.date(),
        "covers_requested_range": actual_start <= expected_start + tolerance and actual_end >= expected_end - tolerance,
        "n_missing_periods": len(missing),
        "critical_gaps": critical_gaps,
    }


def summarize_checks(checks: list[dict]) -> pd.DataFrame:
    """Empilha vários resultados de `check_coverage` numa tabela para inspeção rápida."""
    return pd.DataFrame(checks).set_index("name")
