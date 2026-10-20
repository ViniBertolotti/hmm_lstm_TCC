"""Avaliação pareada de previsões congeladas, sem seleção pelo teste.

Verificações estruturais não autenticam preços nem provam treinamento causal.
O manifesto deve corresponder a todas as janelas e modelos carregados.
"""
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.risk.backtests import (
    acerbi_szekely_unconditional, diebold_mariano, fz0_score, quantile_score, var_coverage,
)

MODELS = ("hmm_lstm", "lstm", "zero_mean_reference")
PAIRS = (("hmm_lstm", "lstm"), ("hmm_lstm", "zero_mean_reference"),
         ("lstm", "zero_mean_reference"))
BLOCKS = ("train", "validation", "calibration", "test")


def _dates(values, label):
    try:
        dates = pd.DatetimeIndex(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Datas inválidas em {label}.") from error
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError(f"{label} exige datas diárias sem NaT ou fuso horário.")
    return dates


def validate_forecasts(frame, manifest, *, sessions):
    """Verifica unidades, cronologia, pareamento e correspondência ao manifesto.

    Exige a mesma amostra nos três modelos e em todos os níveis. Não toma
    silenciosamente a interseção de datas, pois isso pode selecionar resultados.
    Retorna cópia ordenada e número de estados por janela.
    """
    required = {"model", "fold", "origin", "target", "confidence", "regime",
                "actual_log_return", "actual_loss", "var", "es", "residual_pool"}
    if not isinstance(frame, pd.DataFrame) or frame.empty or not required.issubset(frame):
        raise ValueError("Previsões vazias ou sem as colunas necessárias.")
    calendar = _dates(sessions, "calendário")
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("Calendário deve ser não vazio, único e crescente.")
    data = frame.copy()
    for name in ("origin", "target"):
        data[name] = _dates(data[name], name)
    if set(data.model) != set(MODELS):
        raise ValueError("A avaliação requer HMM-LSTM, LSTM e referência de média zero.")
    for name in ("fold", "confidence", "actual_log_return", "actual_loss", "var", "es"):
        try:
            values = data[name].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Valores não numéricos em {name}.") from error
        if not np.isfinite(values).all():
            raise ValueError(f"Valores não finitos em {name}.")
    if ((data.fold < 0) | (data.fold != np.floor(data.fold))).any():
        raise ValueError("Identificadores de janela devem ser inteiros não negativos.")
    if not data.confidence.between(0, 1, inclusive="neither").all():
        raise ValueError("Confiança deve estar em (0,1).")
    if data.duplicated(["model", "confidence", "target"]).any():
        raise ValueError("Previsões duplicadas para o mesmo modelo, nível e alvo.")
    origin_pos, target_pos = calendar.get_indexer(data.origin), calendar.get_indexer(data.target)
    if (origin_pos < 0).any() or (target_pos < 0).any() or not (target_pos-origin_pos == 1).all():
        raise ValueError("Cada alvo deve ser o próximo pregão após sua origem no calendário.")
    if (data.es < data["var"]).any():
        raise ValueError("ES não pode ser inferior ao VaR na convenção de perdas.")
    if not np.allclose(data.actual_loss, -np.expm1(data.actual_log_return), rtol=1e-12, atol=1e-14):
        raise ValueError("Perda não corresponde ao negativo do retorno simples.")
    data = data.sort_values(["model", "confidence", "target"]).reset_index(drop=True)
    levels = sorted(data.confidence.unique())
    reference = data[(data.model == "hmm_lstm") & (data.confidence == levels[0])].set_index("target")
    for model in MODELS:
        for level in levels:
            subset = data[(data.model == model) & (data.confidence == level)].set_index("target")
            if not subset.index.equals(reference.index):
                raise ValueError("Modelos e níveis precisam dos mesmos alvos; datas ausentes não serão descartadas.")
            for name in ("origin", "fold"):
                if not subset[name].equals(reference[name]):
                    raise ValueError(f"{name} diverge entre modelos ou níveis para o mesmo alvo.")
            if not np.allclose(subset[["actual_log_return", "actual_loss"]],
                               reference[["actual_log_return", "actual_loss"]], rtol=1e-12, atol=1e-14):
                raise ValueError("Realizações divergem entre modelos ou níveis para o mesmo alvo.")
    hmm = data[data.model == "hmm_lstm"]
    if hmm.groupby("target").regime.nunique(dropna=False).max() != 1:
        raise ValueError("O regime da origem deve ser o mesmo em todos os níveis.")
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("Manifesto deve ser uma lista não vazia de ajustes.")
    entries = {}
    for item in manifest:
        try:
            key = (item["fold"], item["model"])
        except (TypeError, KeyError) as error:
            raise ValueError("Entrada inválida no manifesto.") from error
        if key in entries:
            raise ValueError("Ajustes duplicados no manifesto.")
        entries[key] = item
    folds = sorted(data.fold.unique())
    expected = {(fold, model) for fold in folds for model in MODELS if model != "zero_mean_reference"}
    if set(entries) != expected:
        raise ValueError("Manifesto e arquivos de previsões cobrem janelas ou modelos diferentes.")
    n_states = {}
    for (fold, model), item in entries.items():
        subset = data[(data.fold == fold) & (data.model == model)]
        try:
            blocks = {b: tuple(_dates([item["block_dates"][b][key] for key in ("first", "last")], b)) for b in BLOCKS}
            count = item["n_sequences"]["test"]
        except (KeyError, TypeError) as error:
            raise ValueError("Manifesto sem blocos temporais e contagem do teste.") from error
        if any(first > last for first, last in blocks.values()):
            raise ValueError("Bloco temporal invertido no manifesto.")
        if any(blocks[a][1] >= blocks[b][0] for a, b in zip(BLOCKS[:-1], BLOCKS[1:])):
            raise ValueError("Blocos do manifesto se sobrepõem ou estão fora de ordem.")
        if subset.target.nunique() != count:
            raise ValueError("Contagem de previsões difere do teste declarado no manifesto.")
        if subset.origin.min() < blocks["test"][0] or subset.target.max() > blocks["test"][1]:
            raise ValueError("Previsão fora do bloco de teste declarado no manifesto.")
        if model == "hmm_lstm":
            states = item.get("n_states")
            if isinstance(states, bool) or not isinstance(states, Integral) or states < 2:
                raise ValueError("Número de estados inválido no manifesto HMM.")
            regimes = subset.regime.to_numpy(dtype=float)
            if (not np.isfinite(regimes).all() or (regimes != np.floor(regimes)).any()
                    or (regimes < 0).any() or (regimes >= states).any()):
                raise ValueError("Regime ausente ou incompatível com o número de estados da janela.")
            n_states[fold] = states
    return data, n_states


def validate_benchmark(benchmark, reference):
    """Benchmark externo (ex.: MS-GARCH) precisa dos mesmos alvos e realizações.

    Não há manifesto de treino Python para ele; a proveniência fica no
    executor que o gerou. Aqui só se garante o pareamento com os outros modelos.
    """
    required = {"model", "fold", "origin", "target", "confidence", "actual_log_return", "actual_loss", "var", "es"}
    if not isinstance(benchmark, pd.DataFrame) or benchmark.empty or not required.issubset(benchmark):
        raise ValueError("Benchmark vazio ou sem as colunas necessárias.")
    data = benchmark.copy()
    for name in ("origin", "target"):
        data[name] = _dates(data[name], name)
    if set(data.model) & set(MODELS):
        raise ValueError("Benchmark não pode reutilizar o nome de um modelo avaliado.")
    if not np.isfinite(data[["var", "es"]].to_numpy(dtype=float)).all() or (data.es < data["var"]).any():
        raise ValueError("Benchmark com VaR/ES não finito ou ES inferior ao VaR.")
    if data.duplicated(["model", "confidence", "target"]).any():
        raise ValueError("Previsões duplicadas no benchmark.")
    data["regime"] = np.nan
    data["residual_pool"] = data.get("residual_pool", "benchmark")
    for model in data.model.unique():
        for level in sorted(reference.confidence.unique()):
            ref = reference[reference.confidence == level].set_index("target").sort_index()
            sub = data[(data.model == model) & (data.confidence == level)].set_index("target").sort_index()
            if not sub.index.equals(ref.index) or not (sub.fold == ref.fold).all() or not (sub.origin == ref.origin).all():
                raise ValueError(f"Benchmark {model} não cobre exatamente os alvos dos outros modelos.")
            if not np.allclose(sub.actual_loss, ref.actual_loss, rtol=1e-12, atol=1e-14):
                raise ValueError(f"Realizações do benchmark {model} divergem dos outros modelos.")
    return data[list(reference.columns.intersection(data.columns))].sort_values(["model", "confidence", "target"])


def coverage_row(subset, sessions, level, null_losses=None):
    if subset.empty:
        return {"n": 0, "es_status": "amostra_vazia"}, {}
    subset = subset.sort_values("target").set_index("target")
    cov = var_coverage(subset.actual_loss, subset["var"], level, sessions=sessions)
    if len(subset) < 2:
        es = {"status": "not_applicable_insufficient_observations"}
    elif (subset.es <= 0).any():
        es = {"status": "not_applicable_es_not_positive"}
    else:
        paths = None
        if null_losses is not None:
            simulated = null_losses.loc[:, subset.index].to_numpy()
            paths = {"losses": simulated, "var": np.broadcast_to(subset["var"].to_numpy(), simulated.shape),
                     "es": np.broadcast_to(subset.es.to_numpy(), simulated.shape)}
        es = acerbi_szekely_unconditional(subset.actual_loss.to_numpy(), subset["var"].to_numpy(), subset.es.to_numpy(),
                                          level, null_paths=paths)
    lo, hi = cov["exact_binomial"]["failure_rate_ci95"]
    row = {"n": cov["n_observations"], "violacoes": cov["n_failures"],
           "esperado": cov["expected_failures_under_nominal_coverage"], "taxa": cov["failure_rate"],
           "taxa_ic95_inferior": lo, "taxa_ic95_superior": hi,
           "binomial_exato_p": cov["exact_binomial"]["p_value_two_sided"],
           "kupiec_p": cov["kupiec"]["p_value_chi2_1"],
           "independencia_p": cov["independence"]["p_value_chi2_1"],
           "cobertura_cond_p": cov["conditional_coverage"]["p_value_chi2_2"],
           "n_transicoes_adjacentes": cov["n_adjacent_valid_transitions"],
           "n_sessoes_sem_par_no_intervalo": cov["n_missing_calendar_sessions"],
           "alerta_contagens_esparsas": "sparse_transition_counts_weaken_chi_square_approximation" in cov["caveats"],
           "es_status": es["status"], "acerbi_szekely_z2": es.get("statistic_z_unconditional"),
           "es_p": es.get("p_value_lower_tail")}
    row.update(cov["transition_counts"])
    return row, {"var": cov, "es": es}


SCORES = {"quantilico_var": lambda s, level: quantile_score(s.actual_loss.to_numpy(), s["var"].to_numpy(), level),
          "fz0_var_es": lambda s, level: fz0_score(s.actual_loss.to_numpy(), s["var"].to_numpy(), s.es.to_numpy(), level)}


def dm_row(frame, a, b, level, sessions, dates, max_lag, score="quantilico_var"):
    if len(dates) < max(10, max_lag+1):
        return {"status": "amostra_insuficiente", "n_pares": len(dates)}, {}
    subsets = {model: frame[(frame.model == model) & (frame.confidence == level)].set_index("target").loc[dates]
               for model in (a, b)}
    excluded = 0
    if score == "fz0_var_es":
        # FZ0 só é definido com ES > 0. Dias em que algum modelo prevê ganho até na cauda
        # saem do par; a regra usa só as previsões, não o resultado.
        valid = (subsets[a].es > 0) & (subsets[b].es > 0)
        excluded = int((~valid).sum())
        subsets = {model: subset[valid.to_numpy()] for model, subset in subsets.items()}
        if len(subsets[a]) < max(10, max_lag+1):
            return {"status": "amostra_insuficiente", "n_pares": len(subsets[a]), "dias_excluidos_es_nao_positivo": excluded}, {}
    scores = {model: pd.Series(SCORES[score](subset, level), index=subset.index) for model, subset in subsets.items()}
    result = diebold_mariano(scores[a], scores[b], sessions=sessions, max_lag=max_lag)
    lower = upper = standard_error = None
    if result["status"] == "computed":
        standard_error = float(np.sqrt(result["long_run_variance"]/result["n_paired"]))
        half_width = float(norm.ppf(.975))*standard_error
        lower, upper = result["mean_loss_a_minus_b"]-half_width, result["mean_loss_a_minus_b"]+half_width
    return {"status": result["status"], "n_pares": len(scores[a]), "dias_excluidos_es_nao_positivo": excluded,
            "media_escore_a": float(scores[a].mean()), "media_escore_b": float(scores[b].mean()),
            "diferenca_a_menos_b": result["mean_loss_a_minus_b"],
            "erro_padrao_hac": standard_error, "diferenca_ic95_inferior": lower, "diferenca_ic95_superior": upper,
            "dm_estatistica": result["statistic"], "dm_p": result["p_value_two_sided"],
            "max_lag": max_lag, "leitura": "diferença positiva favorece b; IC normal assintótico; não é teste de não inferioridade"}, result


def evaluate_forecasts(frame, manifest, *, sessions, dm_max_lag=5, benchmark=None, null_losses=None):
    """Avalia amostra pareada, preservando distâncias do calendário nos segmentos.

    Intervalos binomiais exatos não corrigem dependência temporal; ICs DM são
    assintóticos HAC. Não decide H1–H3 nem inventa margem de não inferioridade.
    `benchmark`: previsões de modelos externos (ex.: MS-GARCH) nos mesmos alvos;
    entram na cobertura e em comparações DM contra cada modelo avaliado.
    `null_losses`: {modelo: DataFrame (simulações × alvos)} com perdas sorteadas
    da distribuição prevista pelo próprio modelo, mantendo VaR/ES fixos; dá o
    p-valor Acerbi–Székely (Z2) unilateral para subestimação.
    """
    if isinstance(dm_max_lag, bool) or not isinstance(dm_max_lag, Integral) or dm_max_lag < 0:
        raise ValueError("Defasagem HAC deve ser um inteiro não negativo.")
    frame, n_states = validate_forecasts(frame, manifest, sessions=sessions)
    models, pairs = list(MODELS), list(PAIRS)
    if benchmark is not None:
        reference = frame[frame.model == "hmm_lstm"]
        extra = validate_benchmark(benchmark, reference)
        frame = pd.concat([frame, extra], ignore_index=True)
        for name in extra.model.unique():
            models.append(name)
            pairs += [(model, name) for model in MODELS]
    null_losses = null_losses or {}
    level0 = frame.confidence.min()
    regime = frame[(frame.model == "hmm_lstm") & (frame.confidence == level0)].set_index("target")
    stress = regime.regime == regime.fold.map(n_states)-1
    segments = {"todos": regime.index, "estresse": regime.index[stress], "demais_regimes": regime.index[~stress]}
    rows, comparisons, details = [], [], []
    for level in sorted(frame.confidence.unique()):
        for segment, dates in segments.items():
            for model in models:
                subset = frame[(frame.model == model) & (frame.confidence == level) & frame.target.isin(dates)]
                row, report = coverage_row(subset, sessions, level, null_losses.get(model))
                identity = {"nivel": float(level), "segmento": segment, "modelo": model}
                rows.append(identity | row); details.append(identity | {"tipo": "cobertura", "resultado": report})
            for a, b in pairs:
                for score in SCORES:
                    row, report = dm_row(frame, a, b, level, sessions, dates, dm_max_lag, score)
                    identity = {"nivel": float(level), "segmento": segment, "a": a, "b": b, "escore": score}
                    comparisons.append(identity | row); details.append(identity | {"tipo": "dm", "resultado": report})
    methods = {model: sorted({m.get("risk_method", "não registrado") for m in manifest if m["model"] == model})
               for model in ("hmm_lstm", "lstm")}
    pools = regime.residual_pool.value_counts(normalize=True).to_dict()
    summary = {"status": "exploratory_frozen_forecasts_structural_checks_passed",
        "periodo_teste": [str(frame.target.min().date()), str(frame.target.max().date())],
        "n_janelas": int(frame.fold.nunique()), "dias_de_teste": len(regime), "dias_em_estresse": int(stress.sum()),
        "niveis_observados": sorted(float(x) for x in frame.confidence.unique()),
        "uso_da_cauda_por_regime": {str(k): float(v) for k, v in pools.items()},
        "metodos_de_risco_registrados": methods,
        "benchmarks": sorted(set(models)-set(MODELS)),
        "es_pvalue_models": sorted(null_losses),
        "H1_H2_status": ("conclusao_final_pendente_validacao_dados" if benchmark is not None
                         else "conclusao_final_pendente_validacao_dados_e_benchmark_MSGARCH"),
        "H3_status": "nao_inferioridade_nao_avaliada_margem_e_procedimento_nao_predefinidos",
        "data_source_validation": "not_established_by_this_evaluator",
        "training_provenance": "manifest_consistency_checked_training_not_reexecuted_or_authenticated",
        "stress_definition": "highest_numeric_state_assumes_ordering_by_training_volatility",
        "dm_max_lag": int(dm_max_lag),
        "limitations": [
            "A correspondência de manifesto e previsões não autentica os preços ou as rolagens.",
            "Esta avaliação não comprova os horários de disponibilidade nem a causalidade de todo o treinamento.",
            ("O benchmark externo é pareado nos mesmos alvos, mas sua proveniência vem do executor que o gerou."
             if benchmark is not None else "O MS-GARCH não integra os três modelos avaliados."),
            "Métodos de calibração diferentes comparam o pipeline completo, sem isolar o efeito da arquitetura.",
            "Contagens e intervalos binomiais usam referência Bernoulli; dependência e amostras raras exigem cautela.",
            "ICs e p-valores DM são assintóticos, não ajustam multiplicidade e dependem da estabilidade das diferenças de perda.",
            ("p-valor A-S sorteia perdas da distribuição prevista em cada dia com VaR/ES fixos (Acerbi e Székely, 2014);"
             " não reproduz a reestimação nem a dependência entre dias." if null_losses
             else "Sem trajetórias sob H0, o ES tem estatística descritiva A-S, sem p-valor."),
            "Não rejeitar uma hipótese não prova equivalência, não inferioridade ou correção do modelo.",
        ]}
    return {"coverage": pd.DataFrame(rows), "comparisons": pd.DataFrame(comparisons),
            "details": details, "summary": summary}
