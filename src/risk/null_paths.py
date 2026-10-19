"""Perdas simuladas sob H0 para o p-valor do teste de ES (Acerbi e Székely, 2014).

Sob H0, a perda de cada dia segue a distribuição que o próprio modelo previu.
Como todas as previsões aqui são simulação histórica filtrada, essa distribuição
é a dos cenários do dia: previsão + escala × erros padronizados. Sorteia-se um
cenário por dia e por trajetória, com VaR e ES mantidos fixos, como no teste Z2
original. Não se reestima o modelo em cada trajetória.
"""
import numpy as np
import pandas as pd

from src.risk.var_es import empirical_var_es, losses_from_log_returns


def sample_scenario_losses(days, *, n_simulations=999, seed=0):
    """`days`: iterável de (alvo, cenários de log-retorno). Retorna simulações × alvos."""
    rng = np.random.default_rng(seed)
    targets, columns = [], []
    for target, scenarios in days:
        scenarios = np.asarray(scenarios, dtype=float)
        if scenarios.ndim != 1 or len(scenarios) < 2 or not np.isfinite(scenarios).all():
            raise ValueError(f"{target}: cenários inválidos.")
        targets.append(pd.Timestamp(target))
        columns.append(losses_from_log_returns(rng.choice(scenarios, size=n_simulations)))
    if pd.DatetimeIndex(targets).has_duplicates:
        raise ValueError("Alvos repetidos nas trajetórias sob H0.")
    return pd.DataFrame(np.column_stack(columns), columns=pd.DatetimeIndex(targets))


def check_reconstruction(scenarios, var, es, confidence, *, tolerance=1e-10):
    """Confirma que os cenários reconstruídos reproduzem o VaR/ES já salvos."""
    risk = empirical_var_es(losses_from_log_returns(scenarios), confidence)
    if abs(risk["var"] - var) > tolerance or abs(risk["es"] - es) > tolerance:
        raise ValueError("Cenários reconstruídos não reproduzem o VaR/ES salvo.")
