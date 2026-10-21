"""Consolida as tabelas finais do TCC em reports/tables/final/ (CSV + uma planilha).

Execute da raiz depois de evaluate_walk_forward.py (carteira) e summarize_univariate.py:
    .venv/Scripts/python scripts/build_final_tables.py

Não recalcula modelos: só reorganiza saídas já geradas, com nomes e unidades legíveis,
para copiar para o texto. As figuras ficam em reports/figures/ (notebooks 04-09).
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports/tables"
WF = TABLES / "walk_forward"
EVAL = WF / "evaluation_v3"
OUT = TABLES / "final"
NAMES = {"hmm_lstm": "HMM-LSTM", "lstm": "LSTM", "zero_mean_reference": "FHS-EWMA", "ms_garch": "MS-GARCH"}
SEGMENTS = {"todos": "Todos os dias", "estresse": "Estresse", "demais_regimes": "Demais regimes"}
ASSETS = {"SLCE3.SA": "SLC Agrícola", "JBSS32.SA": "JBS", "MBRF3.SA": "Marfrig/MBRF", "SMTO3.SA": "São Martinho",
          "ZS=F": "Soja (BRL)", "KC=F": "Café arábica (BRL)", "BRL=X": "USD/BRL"}


def descriptive():
    features = pd.read_parquet(ROOT / "data/processed/features.parquet")
    returns = features.xs("return_1d", axis=1, level=1).rename(columns=ASSETS)
    returns["Carteira"] = pd.read_parquet(ROOT / "data/processed/portfolio_return.parquet").return_1d
    rows = {}
    for name, r in returns.items():
        r = r.dropna()
        rows[name] = {"início": r.index[0].date(), "fim": r.index[-1].date(), "n": len(r),
                      "média anual %": r.mean() * 252 * 100, "vol. anual %": r.std() * np.sqrt(252) * 100,
                      "assimetria": stats.skew(r), "curtose (excesso)": stats.kurtosis(r),
                      "pior dia %": r.min() * 100, "melhor dia %": r.max() * 100}
    return pd.DataFrame(rows).T


def windows():
    manifest = pd.DataFrame(json.loads((WF / "manifest.json").read_text(encoding="utf-8")))
    hmm = manifest[manifest.model == "hmm_lstm"].set_index("fold")
    lstm = manifest[manifest.model == "lstm"].set_index("fold")
    fits = pd.read_parquet(TABLES / "msgarch/msgarch_fits.parquet")
    chosen = fits.loc[fits.groupby("fold").bic.idxmin()].set_index("fold").K
    out = pd.DataFrame({
        "teste início": hmm.block_dates.map(lambda d: d["test"]["first"][:10]),
        "teste fim": hmm.block_dates.map(lambda d: d["test"]["last"][:10]),
        "dias avaliados": hmm.n_sequences.map(lambda d: d["test"]),
        "estados HMM": hmm.n_states, "estados MS-GARCH": chosen,
        "época LSTM": lstm.best_epoch, "época HMM-LSTM": hmm.best_epoch,
        "MSE LSTM / média zero": lstm.point_mse.map(lambda d: d["lstm"] / d["zero_mean_reference"]),
        "MSE HMM-LSTM / média zero": hmm.point_mse.map(lambda d: d["hmm_lstm"] / d["zero_mean_reference"]),
    })
    out.index.name = "janela"
    return out


def coverage(frame):
    t = frame.copy()
    t["modelo"] = t.modelo.map(NAMES)
    t["segmento"] = t.segmento.map(SEGMENTS)
    t["nível"] = (t.nivel * 100).astype(int).astype(str) + "%"
    t["taxa %"] = t.taxa * 100
    t = t.rename(columns={"violacoes": "violações", "esperado": "esperadas", "kupiec_p": "Kupiec p",
                          "independencia_p": "independência p", "cobertura_cond_p": "cob. condicional p",
                          "acerbi_szekely_z2": "Z2 (ES)", "es_p": "ES p"})
    return t[["nível", "segmento", "modelo", "n", "violações", "esperadas", "taxa %", "Kupiec p",
              "independência p", "cob. condicional p", "Z2 (ES)", "ES p"]]


def comparisons(frame):
    t = frame.copy()
    t["a"], t["b"] = t.a.map(NAMES), t.b.map(NAMES)
    t["segmento"] = t.segmento.map(SEGMENTS)
    t["nível"] = (t.nivel * 100).astype(int).astype(str) + "%"
    t["escore"] = t.escore.map({"quantilico_var": "quantílico (VaR)", "fz0_var_es": "FZ0 (VaR+ES)"})
    t["melhor (p<0,05)"] = np.where(t.dm_p < .05, np.where(t.diferenca_a_menos_b > 0, t.b, t.a), "—")
    return t[["nível", "segmento", "escore", "a", "b", "diferenca_a_menos_b", "diferenca_ic95_inferior",
              "diferenca_ic95_superior", "dm_estatistica", "dm_p", "melhor (p<0,05)"]].rename(
        columns={"diferenca_a_menos_b": "a − b", "diferenca_ic95_inferior": "IC95 inf.",
                 "diferenca_ic95_superior": "IC95 sup.", "dm_estatistica": "DM", "dm_p": "p"})


def hypotheses(cov, dm):
    def c(model, seg, level, col):
        return cov.query("modelo == @model and segmento == @seg and nivel == @level")[col].iloc[0]

    def d(seg, level, score):
        r = dm.query("a == 'hmm_lstm' and b == 'ms_garch' and segmento == @seg and nivel == @level "
                     "and escore == @score").iloc[0]
        return r.dm_p, r.diferenca_a_menos_b

    rows = []
    for level in (.95, .99):
        tag = f"{level:.0%}"
        kp = c("hmm_lstm", "todos", level, "kupiec_p")
        (pq, dq), (pf, df) = d("todos", level, "quantilico_var"), d("todos", level, "fz0_var_es")
        rows.append({"hipótese": "H1", "nível": tag,
                     "critério (pré-projeto)": "cobertura compatível (Kupiec) e desempenho ≥ MS-GARCH",
                     "evidência": f"Kupiec p={kp:.2f}; vs MS-GARCH: DM p={pq:.4f} (quantílico), {pf:.4f} (FZ0)",
                     "conclusão": "parcial: cobertura sim; desempenho inferior ao MS-GARCH"
                                  if kp > .05 and pq < .05 and dq > 0 else "ver evidência"})
        h, m = c("hmm_lstm", "estresse", level, "taxa"), c("ms_garch", "estresse", level, "taxa")
        kh, km = c("hmm_lstm", "estresse", level, "kupiec_p"), c("ms_garch", "estresse", level, "kupiec_p")
        rows.append({"hipótese": "H2", "nível": tag,
                     "critério (pré-projeto)": "no estresse, menor erro de cobertura e menos agrupamento que o MS-GARCH",
                     "evidência": f"violações {h:.1%} (Kupiec p={kh:.2f}) vs {m:.1%} (p={km:.2f})",
                     "conclusão": "não sustentada" if abs(h - (1 - level)) >= abs(m - (1 - level))
                                  else "menor desvio observado; agrupamento e diferença formal precisam ser avaliados"})
        (pq, dq), (pf, df) = d("demais_regimes", level, "quantilico_var"), d("demais_regimes", level, "fz0_var_es")
        rows.append({"hipótese": "H3", "nível": tag,
                     "critério (pré-projeto)": "nos regimes calmos, sem perda significativa vs benchmark (DM)",
                     "evidência": f"DM p={pq:.4f} (quantílico), {pf:.4f} (FZ0); HMM-LSTM {'pior' if dq > 0 else 'melhor'}",
                     "conclusão": "rejeitada" if pq < .05 and dq > 0 else "não rejeitada",
                     'interpretação': 'Critério original de diferença pelo DM; não testa não inferioridade econômica.'})
    return pd.DataFrame(rows)


def msgarch_fits():
    fits = pd.read_parquet(TABLES / "msgarch/msgarch_fits.parquet")
    pars = fits[fits.K == 2].set_index("fold").par.map(json.loads).apply(pd.Series)
    out = pd.DataFrame(index=pars.index)
    for k, label in ((1, "baixa vol."), (2, "alta vol.")):
        persistence = pars[f"alpha1_{k}"] + pars[f"beta_{k}"]
        out[f"persistência ({label})"] = persistence
        out[f"vol. incond. anual % ({label})"] = np.sqrt(pars[f"alpha0_{k}"] / (1 - persistence) * 252)
        out[f"ν Student-t ({label})"] = pars[f"nu_{k}"]
    out["P(fica em baixa vol.)"] = pars.P_1_1
    out["P(alta → baixa)"] = pars.P_2_1  # MSGARCH: P_2_1 = P(s_t=1 | s_{t-1}=2)
    bic = fits.pivot(index="fold", columns="K", values="bic")
    out["BIC K=2"], out["BIC K=3"] = bic[2], bic[3]
    out.index.name = "janela"
    return out


def univariate():
    path = TABLES / "univariado/resumo_cobertura.csv"
    if not path.exists():
        return None, None
    cov = pd.read_csv(path)
    dm = pd.read_csv(TABLES / "univariado/resumo_dm.csv")
    rate = cov[cov.segmento.isin(["todos", "estresse"])].assign(
        modelo=lambda t: t.modelo.map(NAMES), ativo=lambda t: t.ativo.map(ASSETS),
        segmento=lambda t: t.segmento.map(SEGMENTS),
        nível=lambda t: (t.nivel * 100).astype(int).astype(str) + "%")
    rate = rate.pivot_table(index=["ativo", "segmento", "nível"], columns="modelo", values="taxa") * 100
    pairs = dm[(dm.escore == "quantilico_var") & (dm.segmento == "todos")].copy()
    pairs["comparação"] = pairs.a.map(NAMES) + " vs " + pairs.b.map(NAMES)
    pairs["melhor (p<0,05)"] = np.where(pairs.dm_p < .05, np.where(pairs.diferenca_a_menos_b > 0,
                                        pairs.b.map(NAMES), pairs.a.map(NAMES)), "—")
    pairs["ativo"] = pairs.ativo.map(ASSETS)
    pairs["nível"] = (pairs.nivel * 100).astype(int).astype(str) + "%"
    pairs = pairs.pivot_table(index=["ativo", "nível"], columns="comparação", values="melhor (p<0,05)", aggfunc="first")
    return rate, pairs


def density():
    directory = WF / "density"
    if not (directory / "resumo.csv").exists():
        return None, None
    summary = pd.read_csv(directory / "resumo.csv")
    summary["modelo"] = summary.modelo.map(NAMES)
    summary["segmento"] = summary.segmento.map(SEGMENTS)
    summary["CRPS médio %"] = summary.crps_medio * 100
    summary = summary.rename(columns={"ks": "KS", "ks_p": "KS p", "berkowitz_p": "Berkowitz p",
                                      "berkowitz_mu": "μ", "berkowitz_sigma": "σ", "berkowitz_rho": "ρ",
                                      "pit_abaixo_5%": "PIT < 0,05", "pit_acima_95%": "PIT > 0,95"})
    summary = summary[["segmento", "modelo", "n", "CRPS médio %", "KS", "KS p", "Berkowitz p", "μ", "σ", "ρ",
                       "PIT < 0,05", "PIT > 0,95"]]
    dm = pd.read_csv(directory / "crps_dm.csv")
    dm["a"], dm["b"] = dm.a.map(NAMES), dm.b.map(NAMES)
    dm["segmento"] = dm.segmento.map(SEGMENTS)
    dm["melhor (p<0,05)"] = np.where(dm.dm_p < .05, np.where(dm.diferenca_a_menos_b > 0, dm.b, dm.a), "—")
    dm["CRPS a %"], dm["CRPS b %"] = dm.crps_a * 100, dm.crps_b * 100
    dm = dm[["segmento", "a", "b", "CRPS a %", "CRPS b %", "dm", "dm_p", "melhor (p<0,05)"]].rename(
        columns={"dm": "DM", "dm_p": "p"})
    return summary, dm


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cov = pd.read_csv(EVAL / "backtest_cobertura.csv")
    dm = pd.read_csv(EVAL / "backtest_comparacao_dm.csv")
    tables = {"T1_estatisticas_descritivas": descriptive(), "T2_janelas_walk_forward": windows(),
              "T3_cobertura_var_es": coverage(cov), "T4_comparacoes_dm": comparisons(dm),
              "T5_hipoteses": hypotheses(cov, dm), "T6_msgarch_parametros": msgarch_fits()}
    pit, crps = density()
    if pit is not None:
        tables["T9_pit_crps"] = pit
        tables["T10_crps_dm"] = crps
    rate, pairs = univariate()
    if rate is not None:
        tables["T7_univariado_violacoes"] = rate
        tables["T8_univariado_dm"] = pairs
    with pd.ExcelWriter(OUT / "tabelas_tcc.xlsx") as excel:
        for name, table in tables.items():
            index = not isinstance(table.index, pd.RangeIndex)
            table.round(4).to_csv(OUT / f"{name}.csv", index=index, encoding="utf-8-sig")
            table.round(4).to_excel(excel, sheet_name=name[:31], index=index)
    print(f"{len(tables)} tabelas em {OUT}")


if __name__ == "__main__":
    main()
