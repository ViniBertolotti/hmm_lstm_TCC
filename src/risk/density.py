"""Avaliação da distribuição prevista inteira: PIT e CRPS.

Todas as previsões do projeto são distribuições empíricas (cenários de log-retorno do
dia). Para cada dia:

- PIT (Diebold, Gunther e Tay, 1998): u = F(y), a probabilidade prevista de um retorno
  menor ou igual ao realizado. Se o modelo está correto, u ~ Uniforme(0,1) e, num
  horizonte de um dia, os u são independentes. Com cenários discretos, empates e valores
  fora da amostra são tratados por sorteio, o que preserva a uniformidade sob H0.
- CRPS (Gneiting e Raftery, 2007): E|X − y| − ½E|X − X'|, na mesma unidade do retorno.
  Menor é melhor; é uma regra de pontuação própria, então pode ser comparado por
  Diebold–Mariano.
"""
import numpy as np
import pandas as pd
from scipy import stats


def pit_ensemble(scenarios, realized, rng):
    """PIT randomizado de um valor realizado contra um conjunto de cenários."""
    x = np.asarray(scenarios, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(realized):
        raise ValueError("Cenários e realização devem ser finitos; pelo menos dois cenários.")
    below = np.count_nonzero(x < realized)
    ties = np.count_nonzero(x == realized)
    return float((below + rng.uniform() * (ties + 1)) / (len(x) + 1))


def crps_ensemble(scenarios, realized):
    """CRPS exato da distribuição empírica dos cenários (pesos iguais)."""
    x = np.sort(np.asarray(scenarios, dtype=float))
    if x.ndim != 1 or len(x) < 1 or not np.isfinite(x).all() or not np.isfinite(realized):
        raise ValueError("Cenários e realização devem ser finitos.")
    n = len(x)
    spread = 2 * np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n**2  # E|X − X'|
    return float(np.mean(np.abs(x - realized)) - spread / 2)


def berkowitz_test(pit, *, sessions=None):
    """Berkowitz (2001): z = Φ⁻¹(u) deve ser N(0,1) i.i.d.

    Ajusta z_t = μ + ρ(z_{t-1} − μ) + σε por máxima verossimilhança condicional e testa
    μ=0, σ=1, ρ=0 conjuntamente (LR com 3 g.l.). Com calendário, usa somente
    pares de sessões adjacentes. Sem calendário, o vetor deve ser contíguo.
    """
    values = np.asarray(pit, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('PIT deve ser um vetor finito entre zero e um.')
    adjacent = np.ones(max(0, len(values) - 1), dtype=bool)
    if sessions is not None:
        if not isinstance(pit, pd.Series) or not isinstance(pit.index, pd.DatetimeIndex):
            raise ValueError('Informe PIT como Series com datas ao utilizar calendário.')
        calendar = pd.DatetimeIndex(sessions)
        for dates in (pit.index, calendar):
            if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
                raise ValueError('Datas devem ser únicas, crescentes e completas.')
        positions = calendar.get_indexer(pit.index)
        if (positions < 0).any():
            raise ValueError('PIT contém data fora do calendário.')
        adjacent = np.diff(positions) == 1
    u = np.clip(values, 1e-10, 1 - 1e-10)
    z = stats.norm.ppf(u)
    if len(z) < 10 or adjacent.sum() < 9:
        raise ValueError("Amostra pequena demais para Berkowitz: exige nove pares adjacentes.")
    y, x = z[1:][adjacent], z[:-1][adjacent]
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    sigma2 = resid.var()
    if sigma2 <= 0:
        raise ValueError('Variância residual nula no teste de Berkowitz.')
    loglik_alt = -0.5 * len(y) * (np.log(2 * np.pi * sigma2) + 1)
    loglik_null = -0.5 * len(y) * np.log(2 * np.pi) - 0.5 * np.sum(y**2)
    lr = float(2 * (loglik_alt - loglik_null))
    rho = float(beta[1])
    return {"mu": float(beta[0] / (1 - rho)) if abs(rho) < 1 else float("nan"), "rho": rho,
            "sigma": float(np.sqrt(sigma2)), "lr": lr, "p_value": float(stats.chi2.sf(lr, 3)),
            "n": int(len(y)), 'n_gap_pairs_excluded': int((~adjacent).sum())}


def ks_uniform(pit):
    """Kolmogorov–Smirnov contra a uniforme. Ignora dependência entre dias; é descritivo."""
    result = stats.kstest(np.asarray(pit, dtype=float), "uniform")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
