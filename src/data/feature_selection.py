"""Diagnóstico exploratório de features (Semana 3, Passada 2).

`stationarity_report` exige uma série por vez (ADF/KPSS assumem ordenação
temporal) — rodar por ativo separadamente, nunca num painel de vários ativos
concatenados. `correlation_matrix`/`vif_filter` não dependem de ordem
temporal e podem rodar pooled (todos os ativos empilhados). Este módulo é um
diagnóstico na amostra completa, não a seleção final de features — essa
entra dentro de cada split walk-forward (Semana 6), para não vazar
informação de teste para a escolha de quais features manter.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity_report(df: pd.DataFrame) -> pd.DataFrame:
    """ADF (H0: raiz unitária/não-estacionária) e KPSS (H0: estacionária) por coluna.

    Uma feature "estacionária" com confiança deve ter ADF p<0.05 (rejeita
    raiz unitária) *e* KPSS p>0.05 (não rejeita estacionariedade) — os dois
    testes têm hipóteses nulas opostas de propósito, para checagem cruzada.
    """
    rows = []
    for col in df.columns:
        series = df[col].dropna()
        if series.empty or series.nunique() <= 1:
            # ex.: BRL=X não tem volume real no Yahoo -> volume_normalized fica todo NaN/constante
            rows.append(
                {
                    "feature": col,
                    "adf_stat": np.nan,
                    "adf_pvalue": np.nan,
                    "adf_rejects_unit_root": False,
                    "kpss_stat": np.nan,
                    "kpss_pvalue": np.nan,
                    "kpss_fails_to_reject_stationary": False,
                }
            )
            continue
        adf_stat, adf_p = adfuller(series, autolag="AIC", result_object=False)[:2]
        with warnings.catch_warnings():
            # KPSS emite InterpolationWarning quando a estatística cai fora da tabela de valores críticos
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p = kpss(series, regression="c", nlags="auto")[:2]
        rows.append(
            {
                "feature": col,
                "adf_stat": adf_stat,
                "adf_pvalue": adf_p,
                "adf_rejects_unit_root": adf_p < 0.05,
                "kpss_stat": kpss_stat,
                "kpss_pvalue": kpss_p,
                "kpss_fails_to_reject_stationary": kpss_p > 0.05,
            }
        )
    report = pd.DataFrame(rows).set_index("feature")
    report["stationary_both_tests"] = report["adf_rejects_unit_root"] & report["kpss_fails_to_reject_stationary"]
    return report


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr()


def _compute_vifs(df: pd.DataFrame) -> pd.Series:
    """VIF de cada coluna, com intercepto na matriz de design (prática padrão)."""
    design = add_constant(df, has_constant="add")
    return pd.Series(
        [variance_inflation_factor(design.values, i) for i in range(1, design.shape[1])],
        index=df.columns,
    )


def vif_filter(df: pd.DataFrame, threshold: float = 10.0) -> tuple[list[str], pd.DataFrame]:
    """Elimina iterativamente a feature de maior VIF até todas ficarem <= `threshold`.

    Retorna (colunas retidas, relatório com todas as features — mantidas ou
    descartadas — e o VIF no momento da decisão).
    """
    remaining = df.dropna()
    dropped_log: list[tuple[str, float]] = []

    while remaining.shape[1] > 1:
        vifs = _compute_vifs(remaining)
        worst_feature, worst_vif = vifs.idxmax(), vifs.max()
        if worst_vif <= threshold:
            break
        dropped_log.append((worst_feature, worst_vif))
        remaining = remaining.drop(columns=worst_feature)

    final_vifs = _compute_vifs(remaining) if remaining.shape[1] > 1 else pd.Series(dtype=float)

    report = pd.DataFrame(
        [{"feature": f, "status": "dropped", "vif": v} for f, v in dropped_log]
        + [{"feature": f, "status": "kept", "vif": final_vifs.get(f)} for f in remaining.columns]
    ).set_index("feature")
    return list(remaining.columns), report
