"""Backtests de previsões já congeladas; não ajustam nem escolhem o modelo.

Convenção: perda positiva, falha se perda > VaR, confiança c, cauda 1-c.
Séries são indexadas pela sessão de realização. O executor deve comprovar que
cada previsão foi emitida antes dela, com os mesmos alvos/unidades por modelo.
"""
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binomtest, chi2, norm, t as student_t


def _confidence(value):
    if not np.isscalar(value) or not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("Confiança deve estar estritamente entre zero e um.")
    return float(value)


def _aligned(left, right, sessions):
    if not isinstance(left, pd.Series) or not isinstance(right, pd.Series):
        raise ValueError("Entradas devem ser Series com sessões de realização.")
    if not left.index.equals(right.index) or len(left) == 0:
        raise ValueError("As duas séries devem ter o mesmo índice não vazio.")
    if not isinstance(left.index, pd.DatetimeIndex):
        raise ValueError("O índice deve ser DatetimeIndex.")
    index = pd.DatetimeIndex(left.index)
    calendar = pd.DatetimeIndex(sessions)
    for dates in (index, calendar):
        if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError("Datas devem ser únicas, crescentes e sem NaT.")
    if len(calendar) == 0 or not index.isin(calendar).all():
        raise ValueError("Sessão fora do calendário informado.")
    values = pd.DataFrame({"left": left, "right": right}, dtype=float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Valores infinitos não são permitidos.")
    # Ausências dentro da janela permanecem no calendário. Não ligamos sessões
    # separadas por uma lacuna como se fossem dias consecutivos.
    calendar = calendar[(calendar >= index.min()) & (calendar <= index.max())]
    return values.reindex(calendar)


def _bernoulli_loglik(n0, n1, probability):
    return float(xlogy(n0, 1-probability) + xlogy(n1, probability))


def var_coverage(losses, forecasts, confidence, *, sessions):
    """Kupiec, binomial exato, independência e cobertura condicional.

    As distribuições qui-quadrado dos LR são assintóticas. Zero falhas ou
    ausência de um estado anterior tornam a independência não identificável:
    esse caso recebe p=None, e não uma aprovação automática.
    """
    confidence = _confidence(confidence)
    paired = _aligned(losses, forecasts, sessions)
    valid = paired.notna().all(axis=1).to_numpy()
    n = int(valid.sum())
    if n == 0:
        raise ValueError("Não há observações pareadas válidas.")
    failures = (paired.left > paired.right).to_numpy(dtype=int)
    x = int(failures[valid].sum())
    tail = 1-confidence
    ll_null = _bernoulli_loglik(n-x, x, tail)
    ll_best = _bernoulli_loglik(n-x, x, x/n)
    lr_uc = max(0.0, 2*(ll_best-ll_null))
    exact = binomtest(x, n, tail)
    interval = exact.proportion_ci(confidence_level=.95, method="exact")
    pair_valid = valid[:-1] & valid[1:]
    codes = 2*failures[:-1][pair_valid] + failures[1:][pair_valid]
    n00, n01, n10, n11 = map(int, np.bincount(codes, minlength=4))
    report = {
        "n_observations": n, "n_missing_calendar_sessions": int((~valid).sum()),
        "n_failures": x, "failure_rate": x/n, "nominal_tail_probability": tail,
        "expected_failures_under_nominal_coverage": n*tail,
        "n_observed_ties_at_var": int((paired.left[valid] == paired.right[valid]).sum()),
        "kupiec": {"statistic": lr_uc, "p_value_chi2_1": float(chi2.sf(lr_uc, 1))},
        "exact_binomial": {"p_value_two_sided": float(exact.pvalue),
                           "failure_rate_ci95": [float(interval.low), float(interval.high)]},
        "transition_counts": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
        "n_adjacent_valid_transitions": n00+n01+n10+n11,
        "caveats": ["p_values_do_not_prove_model_is_correct",
                    "nominal_tail_test_may_be_conservative_for_discrete_or_rounded_predictions",
                    "missingness_must_not_select_observations_based_on_model_performance"],
    }
    row0, row1 = n00+n01, n10+n11
    if row0 == 0 or row1 == 0:
        report["independence"] = {"status": "not_identifiable", "statistic": None, "p_value_chi2_1": None}
        report["conditional_coverage"] = {"status": "not_identifiable", "statistic": None, "p_value_chi2_2": None}
        return report
    probability = (n01+n11)/(row0+row1)
    ll_iid = _bernoulli_loglik(n00+n10, n01+n11, probability)
    ll_markov = _bernoulli_loglik(n00, n01, n01/row0) + _bernoulli_loglik(n10, n11, n11/row1)
    lr_ind = max(0.0, 2*(ll_markov-ll_iid))
    report["independence"] = {"status": "computed_asymptotic", "statistic": lr_ind,
                              "p_value_chi2_1": float(chi2.sf(lr_ind, 1))}
    report["conditional_coverage"] = {"status": "computed_asymptotic", "statistic": lr_uc+lr_ind,
        "p_value_chi2_2": float(chi2.sf(lr_uc+lr_ind, 2))}
    if min(n00, n01, n10, n11) < 5:
        report["caveats"].append("sparse_transition_counts_weaken_chi_square_approximation")
    return report


def quantile_score(losses, var, confidence):
    """Pinball para quantil de perdas c; menor escore é melhor."""
    confidence = _confidence(confidence)
    losses, var = np.asarray(losses, dtype=float), np.asarray(var, dtype=float)
    if losses.ndim != 1 or losses.shape != var.shape or not len(losses):
        raise ValueError("Perdas e VaR devem ter vetores pareados não vazios.")
    if not np.isfinite(losses).all() or not np.isfinite(var).all():
        raise ValueError("Escore exige valores finitos.")
    difference = losses-var
    return (confidence-(losses < var))*difference


def fz0_score(losses, var, es, confidence):
    """Escore conjunto VaR+ES (FZ0; Fissler e Ziegel, 2016; Patton, Ziegel e Chen, 2019).

    Na escala do retorno, Y=-L, v=-VaR, e=-ES e a=1-c:
    S = -1{Y<=v}(v-Y)/(a·e) + v/e + ln(-e) - 1. Menor é melhor. Exige ES>0.
    """
    confidence = _confidence(confidence)
    losses, var, es = (np.asarray(x, dtype=float) for x in (losses, var, es))
    if losses.ndim != 1 or losses.shape != var.shape or losses.shape != es.shape or not len(losses):
        raise ValueError("Perdas, VaR e ES devem ter vetores pareados não vazios.")
    if not np.isfinite(losses).all() or not np.isfinite(var).all() or not np.isfinite(es).all():
        raise ValueError("Escore exige valores finitos.")
    if (es <= 0).any():
        raise ValueError("FZ0 exige ES positivo na convenção de perdas.")
    tail = 1-confidence
    y, v, e = -losses, -var, -es
    hit = (y <= v).astype(float)
    return -hit*(v-y)/(tail*e) + v/e + np.log(-e) - 1


def diebold_mariano(loss_a, loss_b, *, sessions, max_lag, small_sample_correction=False):
    """DM bilateral para perdas comparáveis de uma sessão, com HAC Bartlett.

    d=loss_a-loss_b: média positiva favorece B. As perdas podem ser MSE de
    retornos ou escores quantílicos do mesmo nível, nunca VaR/ES brutos.
    Lacunas mantêm distância de calendário no HAC. A correção HLN de h=1
    só é aceita com amostra contígua. Não resolve não estacionariedade,
    seleção ex post do benchmark, modelos aninhados ou testes múltiplos.
    """
    paired = _aligned(loss_a, loss_b, sessions)
    valid = paired.notna().all(axis=1)
    n = int(valid.sum())
    if n < 3 or isinstance(max_lag, bool) or not isinstance(max_lag, Integral) or not 0 <= max_lag < n:
        raise ValueError("São necessários >=3 pares e 0 <= max_lag < n.")
    if small_sample_correction and not valid.all():
        raise ValueError("Correção HLN requer amostra de sessões contíguas nesta implementação.")
    difference = (paired.left-paired.right).where(valid)
    mean = float(difference.mean())
    # Scores da média, com contribuição zero nas sessões sem observação.
    # Equivale a HAC da regressão d_t=mu+e_t com peso 0 nas lacunas.
    centered = (difference-mean).fillna(0).to_numpy()
    long_run_variance = float(centered @ centered)/n
    for lag in range(1, max_lag+1):
        weight = 1-lag/(max_lag+1)
        long_run_variance += 2*weight*float(centered[lag:] @ centered[:-lag])/n
    report = {"n_paired": n, "n_missing_calendar_sessions": int((~valid).sum()),
              "mean_loss_a_minus_b": mean, "max_lag": int(max_lag), "kernel": "Bartlett",
              "long_run_variance": long_run_variance, "horizon_sessions": 1,
              "positive_mean_favors": "model_b", "small_sample_hln_h1": bool(small_sample_correction),
              "assumptions": ["same_targets_units_horizon_and_loss_function",
                              "out_of_sample_frozen_forecasts",
                              "weakly_dependent_stable_loss_differential",
                              "bandwidth_chosen_before_inspecting_test_p_values"]}
    if long_run_variance <= 0 or np.ptp(difference.dropna().to_numpy()) == 0:
        report.update(status="degenerate_variance", statistic=None, p_value_two_sided=None)
        return report
    statistic = mean/np.sqrt(long_run_variance/n)
    if small_sample_correction:
        statistic *= np.sqrt((n-1)/n)
        pvalue = 2*student_t.sf(abs(statistic), n-1)
    else:
        pvalue = 2*norm.sf(abs(statistic))
    report.update(status="computed", statistic=float(statistic), p_value_two_sided=float(pvalue))
    return report


def acerbi_szekely_unconditional(losses, var, es, confidence, *, null_paths=None):
    """Z=1-mean(L*I[L>VaR]/ES)/(1-c), negativo indica subestimação.

    Sem distribuição de referência não calcula p-valor. null_paths deve ser
    dict com losses/var/es, matrizes (simulações, sessões) sob H0; previsões
    devem ser regeneradas em cada trajetória quando dependem de seus retornos.
    As trajetórias devem reproduzir a dependência e as regras do modelo.
    A função verifica formato/coerência numérica, não autentica essa hipótese.
    """
    confidence = _confidence(confidence)

    def checked(a, b, e):
        a, b, e = [np.asarray(x, dtype=float) for x in (a, b, e)]
        if a.shape != b.shape or a.shape != e.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(e).all():
            raise ValueError("Perdas, VaR e ES devem ter formas iguais e valores finitos.")
        if (e <= 0).any() or (e < b).any():
            raise ValueError("Este teste normalizado exige ES positivo e ES >= VaR.")
        return a, b, e

    losses, var, es = checked(losses, var, es)
    if losses.ndim != 1 or len(losses) < 2:
        raise ValueError("Informe pelo menos duas perdas observadas em vetor.")
    statistic = float(1-np.mean(losses*(losses > var)/es)/(1-confidence))
    report = {"statistic_z_unconditional": statistic, "confidence": confidence,
              "n_observations": len(losses), "n_failures": int((losses > var).sum()),
              "underestimation_direction": "negative", "p_value_lower_tail": None,
              "caveats": ["sensitive_to_var_forecast_errors",
                          "not_a_time_independence_test",
                          "null_paths_must_match_forecast_distribution_and_temporal_dependence"]}
    if null_paths is None:
        report["status"] = "statistic_only_null_distribution_missing"
        return report
    if not {"losses", "var", "es"}.issubset(null_paths):
        raise ValueError("Distribuição nula requer perdas, VaR e ES por trajetória.")
    simulated, simulated_var, simulated_es = checked(*(null_paths[k] for k in ("losses", "var", "es")))
    if simulated.ndim != 2 or simulated.shape[1] != len(losses) or simulated.shape[0] < 99:
        raise ValueError("São necessárias >=99 trajetórias com o mesmo número de sessões.")
    z = 1-np.mean(simulated*(simulated > simulated_var)/simulated_es, axis=1)/(1-confidence)
    count = int((z <= statistic).sum())
    pvalue = (1+count)/(1+len(z))
    report.update(status="simulation_reference_computed", p_value_lower_tail=float(pvalue),
                  n_simulations=len(z), min_possible_p_value=1/(1+len(z)),
                  null_statistic_mean=float(z.mean()),
                  monte_carlo_standard_error_approx=float(np.sqrt(pvalue*(1-pvalue)/len(z))))
    return report
