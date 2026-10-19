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
from scipy.special import logsumexp as _logsumexp
from threadpoolctl import threadpool_limits


class _BatchedEmissionHMM(GaussianHMM):
    """Calcula emissões uma vez por E-step, mantendo os blocos independentes.

    hmmlearn 0.3.x solicita uma fatia por sequência em _do_estep. Reutilizar
    a matriz evita refatorar a mesma covariância para cada bloco bootstrap.
    O cache é descartado antes do M-step; score/filter usam o caminho original.
    """

    def _do_estep(self, X, lengths):
        self._emission_cache = super()._compute_log_likelihood(X)
        self._emission_offset = 0
        try:
            return super()._do_estep(X, lengths)
        finally:
            del self._emission_cache
            del self._emission_offset

    def _compute_log_likelihood(self, X):
        if not hasattr(self, '_emission_cache'):
            return super()._compute_log_likelihood(X)
        start = self._emission_offset
        self._emission_offset += len(X)
        return self._emission_cache[start:self._emission_offset]


def _sequence_lengths(n_obs: int, lengths=None) -> list[int]:
    values = [n_obs] if lengths is None else list(lengths)
    if not values or any(int(n) != n or n <= 0 for n in values) or sum(values) != n_obs:
        raise ValueError("lengths deve conter comprimentos positivos que somam n_obs.")
    return [int(n) for n in values]


def _effectively_converged(model: GaussianHMM) -> bool:
    history = list(model.monitor_.history)
    if len(history) < 2 or not np.isfinite(history).all():
        return False
    gain = history[-1] - history[-2]
    return 0 <= gain < model.tol


@threadpool_limits.wrap(limits=1, user_api="blas")
def fit_gaussian_hmm(
    observations: np.ndarray,
    n_states: int,
    n_init: int = 20,
    n_iter: int = 200,
    random_state: int | None = None,
    lengths: list[int] | None = None,
) -> GaussianHMM:
    """Ajusta um HMM gaussiano (covariância cheia) via Baum-Welch.

    Baum-Welch (EM) só garante ótimo local: roda `n_init` inicializações
    aleatórias e mantém a de maior log-verossimilhança, conforme o
    pré-projeto ("múltiplas inicializações aleatórias para mitigar a
    convergência para ótimos locais").
    """
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 2 or observations.shape[1] < 2 or not np.isfinite(observations).all():
        raise ValueError("HMM exige matriz finita com retorno e volatilidade.")
    lengths = _sequence_lengths(len(observations), lengths)
    rng = np.random.default_rng(random_state)
    best_model, best_score = None, -np.inf

    for _ in range(n_init):
        model = _BatchedEmissionHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        try:
            model.fit(observations, lengths=lengths)
            if not _effectively_converged(model):
                continue
            score = model.score(observations, lengths=lengths)
        except (ValueError, np.linalg.LinAlgError):
            # covariância singular numa inicialização específica (ex.: estado quase vazio
            # com poucos estados de sobra, comum ao testar n_states acima do real) — descarta
            # essa tentativa e segue para a próxima inicialização aleatória.
            continue
        if np.isfinite(score) and score > best_score:
            best_model, best_score = model, score

    if best_model is None:
        raise RuntimeError(f"Nenhuma das {n_init} inicializações convergiu para n_states={n_states}.")
    return best_model


def select_n_states(
    observations: np.ndarray,
    state_range: tuple[int, ...] = (2, 3, 4),
    n_init: int = 20,
    random_state: int | None = None,
    lengths: list[int] | None = None,
) -> pd.DataFrame:
    """Busca em grade do número de estados, com seleção por BIC/AIC.

    A escolha final ainda pede "inspeção visual da plausibilidade econômica"
    (pré-projeto) — este relatório é insumo para essa decisão humana, não a
    decide sozinho.
    """
    n_obs = len(observations)
    rows = []
    for n_states in state_range:
        model = fit_gaussian_hmm(observations, n_states, n_init=n_init, random_state=random_state, lengths=lengths)
        log_likelihood = model.score(observations, lengths=lengths)
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
    """Médias e persistência de cada estado, ordenados por volatilidade."""
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


def viterbi_decode(model: GaussianHMM, observations: np.ndarray, lengths=None) -> np.ndarray:
    """Sequência de estados mais provável (in-sample) — usa toda a série, inclusive o futuro
    de cada ponto. Serve para caracterizar regimes, não para gerar a feature do LSTM."""
    order = _state_order_by_volatility(model)
    remap = np.argsort(order)  # remap[estado_original] = rank por volatilidade
    _, state_sequence = model.decode(observations, algorithm="viterbi", lengths=lengths)
    return remap[state_sequence]


def forward_filter(model: GaussianHMM, observations: np.ndarray, lengths=None) -> np.ndarray:
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
    lengths = _sequence_lengths(n_obs, lengths)
    starts = set(np.cumsum([0] + lengths[:-1]).tolist())
    log_emission = model._compute_log_likelihood(observations)  # (n_obs, n_states)
    with np.errstate(divide="ignore"):  # log(0) = -inf é o valor correto p/ estado de prob. zero
        log_startprob = np.log(model.startprob_)
        log_transmat = np.log(model.transmat_)

    log_alpha = np.empty((n_obs, n_states))
    log_alpha[0] = log_startprob + log_emission[0]
    log_alpha[0] -= _logsumexp(log_alpha[0])

    for t in range(1, n_obs):
        if t in starts:
            log_alpha[t] = log_startprob + log_emission[t]
            log_alpha[t] -= _logsumexp(log_alpha[t])
            continue
        for j in range(n_states):
            log_alpha[t, j] = _logsumexp(log_alpha[t - 1] + log_transmat[:, j]) + log_emission[t, j]
        log_alpha[t] -= _logsumexp(log_alpha[t])

    order = _state_order_by_volatility(model)
    return np.exp(log_alpha)[:, order]


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
    lengths: list[int] | None = None,
) -> dict:
    """Robustez do HMM via bootstrap em blocos (pré-projeto, seção 3.3).

    Reamostra blocos sem atravessar lacunas e ajusta cada bloco como sequência
    independente. Compara os modelos na cronologia original. Concordância direta
    usa apenas o mesmo número de estados; ARI inclui todos os ajustes bem-sucedidos.
    Registra falhas; as métricas são condicionais aos ajustes bem-sucedidos.
    """
    rng = np.random.default_rng(random_state)
    n_obs = len(observations)
    lengths = _sequence_lengths(n_obs, lengths)

    reference_n_states = select_n_states(observations, state_range, n_init, random_state, lengths=lengths)["bic"].idxmin()
    reference_model = fit_gaussian_hmm(observations, n_states=reference_n_states, n_init=n_init, random_state=random_state, lengths=lengths)
    reference_viterbi = viterbi_decode(reference_model, observations, lengths=lengths)

    selected_states = []
    agreement_pcts, aris = [], []

    for _ in range(n_bootstrap):
        offsets = np.cumsum([0] + lengths[:-1])
        valid_starts = np.concatenate([np.arange(o, o + n - block_length + 1) for o, n in zip(offsets, lengths) if n >= block_length])
        if not len(valid_starts):
            raise ValueError("Nenhuma sequência comporta block_length.")
        starts = rng.choice(valid_starts, size=int(np.ceil(n_obs / block_length)))
        boot_idx = np.concatenate([np.arange(s, s + block_length) for s in starts])[:n_obs]
        boot_obs = observations[boot_idx]
        boot_lengths = [block_length] * (n_obs // block_length)
        if n_obs % block_length:
            boot_lengths.append(n_obs % block_length)
        try:
            boot_seed = int(rng.integers(0, 2**31 - 1))
            bic_scores = select_n_states(boot_obs, state_range, n_init, boot_seed, lengths=boot_lengths)["bic"]
            n_states_boot = bic_scores.idxmin()
            boot_model = fit_gaussian_hmm(boot_obs, n_states_boot, n_init=n_init, random_state=boot_seed, lengths=boot_lengths)
        except RuntimeError:
            continue
        selected_states.append(n_states_boot)
        # Avalia os modelos na MESMA cronologia original, sem escolher uma
        # ocorrência arbitrária de datas duplicadas da reamostragem.
        boot_labels = viterbi_decode(boot_model, observations, lengths=lengths)
        agreement_pcts.append((reference_viterbi == boot_labels).mean() if n_states_boot == reference_n_states else np.nan)
        aris.append(adjusted_rand_score(reference_viterbi, boot_labels))

    if not aris:
        raise RuntimeError("Nenhum ajuste bootstrap convergiu; não há estabilidade estimável.")
    comparable = np.asarray(agreement_pcts)[np.isfinite(agreement_pcts)]
    return {
        "reference_n_states": reference_model.n_components,
        "selected_n_states_distribution": pd.Series(selected_states).value_counts(normalize=True).sort_index(),
        "regime_agreement_pct_mean": float(np.mean(comparable)) if len(comparable) else np.nan,
        "regime_agreement_pct_std": float(np.std(comparable)) if len(comparable) else np.nan,
        "adjusted_rand_index_mean": float(np.mean(aris)),
        "adjusted_rand_index_std": float(np.std(aris)),
        "n_bootstrap_completed": len(agreement_pcts),
        "n_bootstrap_failed": n_bootstrap - len(agreement_pcts),
        "n_bootstrap_same_n_states": len(comparable),
    }
