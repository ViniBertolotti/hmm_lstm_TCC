"""VaR/ES da distribuição empírica de perdas, incluindo átomos no quantil."""
import numpy as np


def losses_from_log_returns(log_returns):
    """Perda como fração do capital: L=1-exp(r), sem truncar ganhos em zero."""
    values = np.asarray(log_returns, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Cenários de log-retorno devem ser vetor finito e não vazio.")
    with np.errstate(over="raise", invalid="raise"):
        try:
            return -np.expm1(values)
        except FloatingPointError as exc:
            raise ValueError("Conversão de log-retorno excede a precisão numérica.") from exc


def empirical_var_es(losses, confidence):
    """Quantil inferior da CDF empírica e média da cauda de massa 1-confidence.

    ES=q+E[(L-q)+]/(1-confidence). Inclui a fração necessária do átomo em q,
    em vez de tirar a média de todos os valores maiores ou iguais ao quantil.
    Pesos iguais. Valores negativos representam ganhos e são preservados.
    """
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Perdas devem ser vetor finito e não vazio.")
    if not np.isscalar(confidence) or not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("Confiança deve estar estritamente entre zero e um.")
    quantile = float(np.quantile(values, confidence, method="inverted_cdf"))
    es = quantile + float(np.maximum(values - quantile, 0).mean()) / (1 - confidence)
    return {"var": quantile, "es": es, "confidence": float(confidence),
            "n_scenarios": len(values), "tail_mass_observation_equivalent": len(values) * (1-confidence)}
