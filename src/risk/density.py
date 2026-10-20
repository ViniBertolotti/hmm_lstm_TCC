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


def berkowitz_test(pit):
    """Berkowitz (2001): z = Φ⁻¹(u) deve ser N(0,1) i.i.d.

    Ajusta z_t = μ + ρ(z_{t-1} − μ) + σε por máxima verossimilhança condicional e testa
    μ=0, σ=1, ρ=0 conjuntamente (LR com 3 g.l.). Assume os dias consecutivos da amostra;
    lacunas de calendário são ignoradas.
    """
    u = np.clip(np.asarray(pit, dtype=float), 1e-10, 1 - 1e-10)
    z = stats.norm.ppf(u)
    if len(z) < 10:
        raise ValueError("Amostra pequena demais para o teste de Berkowitz.")
    y, x = z[1:], z[:-1]
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    sigma2 = resid.var()
    loglik_alt = -0.5 * len(y) * (np.log(2 * np.pi * sigma2) + 1)
    loglik_null = -0.5 * len(y) * np.log(2 * np.pi) - 0.5 * np.sum(y**2)
    lr = float(2 * (loglik_alt - loglik_null))
    rho = float(beta[1])
    return {"mu": float(beta[0] / (1 - rho)) if abs(rho) < 1 else float("nan"), "rho": rho,
            "sigma": float(np.sqrt(sigma2)), "lr": lr, "p_value": float(stats.chi2.sf(lr, 3)),
            "n": int(len(y))}


def ks_uniform(pit):
    """Kolmogorov–Smirnov contra a uniforme. Ignora dependência entre dias; é descritivo."""
    result = stats.kstest(np.asarray(pit, dtype=float), "uniform")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
