"""Detecção de regimes de mercado via Hidden Markov Model gaussiano (Semana 4).

Especificação vem do pré-projeto (`docs/pre_projeto_original.docx`, seção 3.3):
HMM gaussiano sobre observações multivariadas (retorno agregado da carteira +
medidas de volatilidade realizada), 2-4 estados via BIC/AIC, Baum-Welch com
múltiplas inicializações, Viterbi para o conjunto de treino (caracterização),
Forward para inferência prospectiva (sem lookahead), robustez via bootstrap
em blocos. Esta é a versão de fit único/exploratório da Semana 4 — o refit
walk-forward de verdade (para não vazar informação futura para o LSTM) é
escopo da Semana 6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import adjusted_rand_score


def fit_gaussian_hmm(
    observations: np.ndarray,
    n_states: int,
    n_init: int = 20,
    n_iter: int = 200,
    random_state: int | None = None,
) -> GaussianHMM:
    """Ajusta um HMM gaussiano (covariância cheia) via Baum-Welch.

    Baum-Welch (EM) só garante ótimo local: roda `n_init` inicializações
    aleatórias e mantém a de maior log-verossimilhança, conforme o
    pré-projeto ("múltiplas inicializações aleatórias para mitigar a
    convergência para ótimos locais").
    """
    rng = np.random.default_rng(random_state)
    best_model, best_score = None, -np.inf

    for _ in range(n_init):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        try:
            model.fit(observations)
        except (ValueError, np.linalg.LinAlgError):
            # covariância singular numa inicialização específica (ex.: estado quase vazio
            # com poucos estados de sobra, comum ao testar n_states acima do real) — descarta
            # essa tentativa e segue para a próxima inicialização aleatória.
            continue
        if not model.monitor_.converged:
            continue
        score = model.score(observations)
        if score > best_score:
            best_model, best_score = model, score

    if best_model is None:
        raise RuntimeError(f"Nenhuma das {n_init} inicializações convergiu para n_states={n_states}.")
    return best_model


def select_n_states(
    observations: np.ndarray,
    state_range: tuple[int, ...] = (2, 3, 4),
    n_init: int = 20,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Busca em grade do número de estados, com seleção por BIC/AIC.

    A escolha final ainda pede "inspeção visual da plausibilidade econômica"
    (pré-projeto) — este relatório é insumo para essa decisão humana, não a
    decide sozinho.
    """
    n_obs = len(observations)
    rows = []
    for n_states in state_range:
        model = fit_gaussian_hmm(observations, n_states, n_init=n_init, random_state=random_state)
        log_likelihood = model.score(observations)
        n_params = (
            n_states * (n_states - 1)  # transição (linhas somam 1)
            + (n_states - 1)  # distribuição inicial
            + n_states * observations.shape[1]  # médias
            + n_states * observations.shape[1] * (observations.shape[1] + 1) // 2  # covariâncias (cheias, simétricas)
        )
        bic = -2 * log_likelihood + n_params * np.log(n_obs)
        aic = -2 * log_likelihood + 2 * n_params
        rows.append({"n_states": n_states, "log_likelihood": log_likelihood, "bic": bic, "aic": aic})
    return pd.DataFrame(rows).set_index("n_states")


def _state_order_by_volatility(model: GaussianHMM, vol_dim: int = 1) -> np.ndarray:
    """Ordena os estados por volatilidade média crescente (dimensão `vol_dim`).

    Resolve label switching de forma economicamente interpretável — estado 0
    = regime mais calmo, estado N-1 = mais estressado — conforme os "regimes
    esperados" descritos no pré-projeto (baixa vol./tendência altista vs.
    alta vol./estresse).
    """
    return np.argsort(model.means_[:, vol_dim])


def characterize_regimes(model: GaussianHMM, observations: np.ndarray, column_names: list[str]) -> pd.DataFrame:
    """Média, covariância e persistência de cada estado, ordenados por volatilidade."""
    order = _state_order_by_volatility(model)
    rows = []
    for rank, state in enumerate(order):
        persistence = model.transmat_[state, state]
        expected_duration = 1 / (1 - persistence) if persistence < 1 else np.inf
        row = {"state_rank": rank, "persistence": persistence, "expected_duration_days": expected_duration}
        for j, col in enumerate(column_names):
            row[f"mean_{col}"] = model.means_[state, j]
        rows.append(row)
    return pd.DataFrame(rows).set_index("state_rank")


def viterbi_decode(model: GaussianHMM, observations: np.ndarray) -> np.ndarray:
    """Sequência de estados mais provável (in-sample) — usa toda a série, inclusive o futuro
    de cada ponto. Serve para caracterizar regimes, não para gerar a feature do LSTM."""
    order = _state_order_by_volatility(model)
    remap = np.argsort(order)  # remap[estado_original] = rank por volatilidade
    _, state_sequence = model.decode(observations, algorithm="viterbi")
    return remap[state_sequence]


def forward_filter(model: GaussianHMM, observations: np.ndarray) -> np.ndarray:
    """P(estado_t | obs_1..t) — filtragem causal (algoritmo Forward), sem lookahead.

    `hmmlearn` não expõe publicamente o forward puro: `predict_proba`/`decode`
    fazem *smoothing* (forward-backward), que usa observações futuras para
    reestimar o passado. A recursão abaixo é o algoritmo Forward padrão
    (Rabiner, 1989), calculado uma vez para toda a série — matematicamente,
    a probabilidade filtrada no instante t depende só de obs_1..t, não
    importa quantas linhas a mais existam no array de entrada.
    """
    n_obs = len(observations)
    n_states = model.n_components
    log_emission = model._compute_log_likelihood(observations)  # (n_obs, n_states)
    with np.errstate(divide="ignore"):  # log(0) = -inf é o valor correto p/ estado de prob. zero
        log_startprob = np.log(model.startprob_)
        log_transmat = np.log(model.transmat_)

    log_alpha = np.empty((n_obs, n_states))
    log_alpha[0] = log_startprob + log_emission[0]
    log_alpha[0] -= _logsumexp(log_alpha[0])

    for t in range(1, n_obs):
        for j in range(n_states):
            log_alpha[t, j] = _logsumexp(log_alpha[t - 1] + log_transmat[:, j]) + log_emission[t, j]
        log_alpha[t] -= _logsumexp(log_alpha[t])

    order = _state_order_by_volatility(model)
    return np.exp(log_alpha)[:, order]


def _logsumexp(x: np.ndarray) -> float:
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def _moving_block_bootstrap_indices(n_obs: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Índices originais (com repetição) de uma reamostragem por blocos móveis,
    concatenando blocos de `block_length` observações consecutivas até cobrir `n_obs`."""
    n_blocks = int(np.ceil(n_obs / block_length))
    starts = rng.integers(0, n_obs - block_length + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_length) for s in starts])
    return idx[:n_obs]


def block_bootstrap_stability(
    observations: np.ndarray,
    state_range: tuple[int, ...] = (2, 3, 4),
    block_length: int = 21,
    n_bootstrap: int = 200,
    n_init: int = 5,
    random_state: int | None = None,
) -> dict:
    """Robustez do HMM via bootstrap em blocos (pré-projeto, seção 3.3).

    Mede duas dimensões de estabilidade:
    - **estabilidade do nº de estados**: distribuição da escolha por BIC em cada reamostragem;
    - **estabilidade da sequência de regimes**: cada reamostragem é decodificada (Viterbi),
      remapeada para os índices de tempo originais (um bloco reamostrado "aponta" para onde
      veio na série original) e comparada ao Viterbi da amostra original nos pontos
      efetivamente cobertos pela reamostragem — % de concordância direta e Adjusted Rand
      Index (ARI, invariante a troca de rótulo, robusto mesmo sem o alinhamento por
      volatilidade já aplicado em `viterbi_decode`).

    Como o bootstrap embaralha a ordem temporal, cada reamostragem cobre só um
    subconjunto dos instantes originais (alguns pontos saem em vários blocos, outros
    em nenhum) — daí a necessidade do remapeamento antes de comparar.
    """
    rng = np.random.default_rng(random_state)
    n_obs = len(observations)

    reference_n_states = select_n_states(observations, state_range, n_init, random_state)["bic"].idxmin()
    reference_model = fit_gaussian_hmm(observations, n_states=reference_n_states, n_init=n_init, random_state=random_state)
    reference_viterbi = viterbi_decode(reference_model, observations)

    selected_states = []
    agreement_pcts, aris = [], []

    for _ in range(n_bootstrap):
        boot_idx = _moving_block_bootstrap_indices(n_obs, block_length, rng)
        boot_obs = observations[boot_idx]

        bic_scores = select_n_states(boot_obs, state_range, n_init, int(rng.integers(0, 2**31 - 1)))["bic"]
        n_states_boot = bic_scores.idxmin()
        selected_states.append(n_states_boot)

        try:
            boot_model = fit_gaussian_hmm(boot_obs, n_states_boot, n_init=n_init, random_state=int(rng.integers(0, 2**31 - 1)))
        except RuntimeError:
            continue
        boot_viterbi = viterbi_decode(boot_model, boot_obs)  # já ordenado por volatilidade

        # remapeia cada posição bootstrap para o índice de tempo original de onde veio;
        # se o mesmo instante original aparece várias vezes, usa o rótulo da última ocorrência
        original_time_labels = pd.Series(boot_viterbi, index=boot_idx)
        original_time_labels = original_time_labels[~original_time_labels.index.duplicated(keep="last")]
        original_time_labels = original_time_labels.sort_index()

        covered = original_time_labels.index.values
        ref_labels = reference_viterbi[covered]
        boot_labels = original_time_labels.values

        agreement_pcts.append((ref_labels == boot_labels).mean())
        aris.append(adjusted_rand_score(ref_labels, boot_labels))

    return {
        "reference_n_states": reference_model.n_components,
        "selected_n_states_distribution": pd.Series(selected_states).value_counts(normalize=True).sort_index(),
        "regime_agreement_pct_mean": float(np.mean(agreement_pcts)),
        "regime_agreement_pct_std": float(np.std(agreement_pcts)),
        "adjusted_rand_index_mean": float(np.mean(aris)),
        "adjusted_rand_index_std": float(np.std(aris)),
        "n_bootstrap_completed": len(agreement_pcts),
    }
