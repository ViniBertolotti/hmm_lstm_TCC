from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

from src.risk.backtests import quantile_score
from src.risk.evaluation import MODELS, evaluate_forecasts, validate_forecasts


@pytest.fixture
def experiment():
    sessions = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(1204)
    losses = rng.normal(0, .012, len(sessions))
    rows, manifest = [], []
    for fold, start, end, states in [(0, 30, 60, 2), (1, 60, 90, 3)]:
        for model in MODELS:
            for level in (.95, .99):
                for j, target_position in enumerate(range(start+1, end+1)):
                    loss = losses[target_position]
                    var = (.016 if level == .95 else .025)+MODELS.index(model)*.002*np.sin(j*.7)
                    rows.append(dict(model=model, fold=fold, origin=sessions[target_position-1],
                        target=sessions[target_position], confidence=level,
                        regime=(j//3)%states if model == "hmm_lstm" else np.nan,
                        var=var, es=var+.008, actual_loss=loss, actual_log_return=np.log1p(-loss),
                        residual_pool="all"))
            if model != "zero_mean_reference":
                intervals = {"train": (0, start-21), "validation": (start-20, start-11),
                             "calibration": (start-10, start-1), "test": (start, end)}
                manifest.append(dict(fold=fold, model=model, n_states=states if model == "hmm_lstm" else None,
                    n_sequences={"test": end-start},
                    risk_method="fhs_regime" if model == "hmm_lstm" else "fhs_global",
                    block_dates={key: {"first": str(sessions[a]), "last": str(sessions[b])}
                                 for key, (a, b) in intervals.items()}))
    return pd.DataFrame(rows), manifest, sessions


def test_evaluation_is_order_invariant_and_keeps_regime_calendar_distances(experiment):
    frame, manifest, sessions = experiment
    original = frame.copy(deep=True)
    first = evaluate_forecasts(frame, manifest, sessions=sessions)
    shuffled = evaluate_forecasts(frame.sample(frac=1, random_state=4), list(reversed(manifest)), sessions=sessions)
    pd.testing.assert_frame_equal(first["coverage"], shuffled["coverage"])
    pd.testing.assert_frame_equal(first["comparisons"], shuffled["comparisons"])
    pd.testing.assert_frame_equal(frame, original)
    assert first["summary"]["dias_de_teste"] == 60
    assert "nao_inferioridade_nao_avaliada" in first["summary"]["H3_status"]
    hmm = frame[(frame.model == "hmm_lstm") & (frame.confidence == .95)]
    stress = hmm[hmm.regime == hmm.fold.map({0: 1, 1: 2})]
    expected_adjacent = int((np.diff(sessions.get_indexer(stress.target)) == 1).sum())
    reported = first["coverage"].query("modelo == 'hmm_lstm' and nivel == .95 and segmento == 'estresse'").iloc[0]
    assert reported.n_transicoes_adjacentes == expected_adjacent < len(stress)-1
    assert reported.es_p is None
    details = [x for x in first["details"] if x["tipo"] == "cobertura" and x["resultado"]]
    assert all(x["resultado"]["es"].get("p_value_lower_tail") is None for x in details)


def test_score_interval_uses_calendar_hac_with_an_independent_regression_oracle(experiment):
    frame, manifest, sessions = experiment
    result = evaluate_forecasts(frame, manifest, sessions=sessions, dm_max_lag=3)
    comparison = result["comparisons"].query("a == 'hmm_lstm' and b == 'lstm' and nivel == .95 and segmento == 'estresse' and escore == 'quantilico_var'").iloc[0]
    indexed = {m: frame[(frame.model == m) & (frame.confidence == .95)].set_index("target") for m in MODELS}
    stress = indexed["hmm_lstm"].regime == indexed["hmm_lstm"].fold.map({0: 1, 1: 2})
    dates = indexed["hmm_lstm"].index[stress]
    score = {m: quantile_score(indexed[m].loc[dates].actual_loss, indexed[m].loc[dates, "var"], .95)
             for m in ("hmm_lstm", "lstm")}
    full = pd.Series(score["hmm_lstm"]-score["lstm"], index=dates).reindex(sessions)
    # Intercept with zero-weight observations retains actual calendar distances.
    weights = full.notna().to_numpy(dtype=float)
    oracle = sm.OLS(full.fillna(0).to_numpy(), weights[:, None]).fit()
    expected_se = np.sqrt(cov_hac(oracle, nlags=3, use_correction=False)[0, 0])
    np.testing.assert_allclose(comparison.erro_padrao_hac, expected_se, rtol=1e-12)
    assert comparison.diferenca_ic95_inferior < comparison.diferenca_a_menos_b < comparison.diferenca_ic95_superior
    np.testing.assert_allclose((comparison.diferenca_ic95_superior-comparison.diferenca_ic95_inferior)/2,
                               1.959963984540054*expected_se, rtol=1e-12)


@pytest.mark.parametrize("case,match", [
    ("missing_manifest", "Manifesto e arquivos"),
    ("extra_manifest", "Manifesto e arquivos"),
    ("duplicate_manifest", "duplicados no manifesto"),
    ("missing_date", "mesmos alvos"),
    ("different_actual", "Realizações divergem"),
    ("nonfinite", "não finitos"),
    ("invalid_horizon", "próximo pregão"),
    ("duplicate_forecast", "Previsões duplicadas"),
    ("overlapping_blocks", "se sobrepõem"),
    ("count", "Contagem"),
    ("state_count", "Regime ausente ou incompatível"),
    ("state_level", "mesmo em todos os níveis"),
    ("es_below_var", "ES não pode"),
    ("loss_units", "Perda não corresponde"),
])
def test_incompatible_experiments_are_rejected_instead_of_silently_scored(experiment, case, match):
    frame, manifest, sessions = experiment
    frame, manifest = frame.copy(), deepcopy(manifest)
    if case == "missing_manifest": manifest.pop()
    elif case == "extra_manifest": manifest.append(dict(manifest[-1], fold=99))
    elif case == "duplicate_manifest": manifest.append(deepcopy(manifest[-1]))
    elif case == "missing_date": frame = frame.drop(frame.index[0])
    elif case == "different_actual":
        frame.loc[0, "actual_loss"] += .01
        frame.loc[0, "actual_log_return"] = np.log1p(-frame.loc[0, "actual_loss"])
    elif case == "nonfinite": frame.loc[0, "var"] = np.inf
    elif case == "invalid_horizon": frame.loc[0, "origin"] = sessions[29]
    elif case == "duplicate_forecast": frame = pd.concat([frame, frame.iloc[:1]])
    elif case == "overlapping_blocks": manifest[0]["block_dates"]["calibration"]["last"] = str(sessions[30])
    elif case == "count": manifest[0]["n_sequences"]["test"] -= 1
    elif case == "state_count": manifest[2]["n_states"] = 2
    elif case == "state_level": frame.loc[0, "regime"] = 1
    elif case == "es_below_var": frame.loc[0, "es"] = frame.loc[0, "var"]-.1
    elif case == "loss_units": frame.loc[0, "actual_loss"] = .5
    with pytest.raises(ValueError, match=match):
        validate_forecasts(frame, manifest, sessions=sessions)


@pytest.mark.parametrize("lag", [-1, True, 1.5])
def test_bandwidth_must_be_an_explicit_nonnegative_integer(experiment, lag):
    frame, manifest, sessions = experiment
    with pytest.raises(ValueError, match="Defasagem HAC"):
        evaluate_forecasts(frame, manifest, sessions=sessions, dm_max_lag=lag)


def _benchmark(frame, shift=.003):
    bench = frame[frame.model == "hmm_lstm"].copy()
    bench["model"], bench["var"], bench["es"] = "ms_garch", bench["var"]+shift, bench["es"]+shift
    return bench.drop(columns=["regime", "residual_pool"])


def test_benchmark_is_paired_and_compared_against_each_model(experiment):
    frame, manifest, sessions = experiment
    result = evaluate_forecasts(frame, manifest, sessions=sessions, benchmark=_benchmark(frame))
    assert result["summary"]["benchmarks"] == ["ms_garch"]
    assert set(result["coverage"].modelo) == set(MODELS) | {"ms_garch"}
    pairs = result["comparisons"][result["comparisons"].b == "ms_garch"]
    assert set(pairs.a) == set(MODELS)
    with pytest.raises(ValueError, match="não cobre exatamente"):
        evaluate_forecasts(frame, manifest, sessions=sessions, benchmark=_benchmark(frame).iloc[1:])


def test_es_pvalue_from_paths_drawn_under_the_model_distribution(experiment):
    from src.risk.null_paths import sample_scenario_losses
    frame, manifest, sessions = experiment
    targets = sorted(frame.target.unique())
    rng = np.random.default_rng(0)
    # Cenários muito mais estreitos que as perdas reais: o ES subestima o risco.
    narrow = sample_scenario_losses([(t, rng.normal(0, .002, 500)) for t in targets], n_simulations=199, seed=1)
    result = evaluate_forecasts(frame, manifest, sessions=sessions, null_losses={"lstm": narrow})
    row = result["coverage"].query("modelo == 'lstm' and nivel == .95 and segmento == 'todos'").iloc[0]
    assert row.es_status == "simulation_reference_computed"
    assert row.es_p <= .01 and row.acerbi_szekely_z2 < 0
    other = result["coverage"].query("modelo == 'hmm_lstm' and nivel == .95 and segmento == 'todos'").iloc[0]
    assert other.es_p is None or np.isnan(other.es_p)


def test_fz0_skips_days_with_non_positive_es_and_reports_them(experiment):
    frame, manifest, sessions = experiment
    frame = frame.copy()
    hit = (frame.model == "lstm") & (frame.confidence == .95) & (frame.target == frame.target.min())
    frame.loc[hit, ["var", "es"]] = [-.02, -.01]  # previsão de ganho até na cauda
    result = evaluate_forecasts(frame, manifest, sessions=sessions)
    row = result["comparisons"].query("a == 'hmm_lstm' and b == 'lstm' and nivel == .95 "
                                      "and segmento == 'todos' and escore == 'fz0_var_es'").iloc[0]
    assert row.dias_excluidos_es_nao_positivo == 1 and row.n_pares == 59
    quantile = result["comparisons"].query("a == 'hmm_lstm' and b == 'lstm' and nivel == .95 "
                                           "and segmento == 'todos' and escore == 'quantilico_var'").iloc[0]
    assert quantile.n_pares == 60
