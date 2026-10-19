"""Montagem do dataset de modelagem em janelas walk-forward (Semana 6).

Regra central: tudo o que é estimado a partir dos dados — seleção de features
(missing + VIF), padronização e o HMM (inclusive o número de estados) — é
ajustado somente com o treino de cada janela. Validação e teste só recebem as
transformações já ajustadas. Assim nenhuma informação do período de teste
influencia as escolhas feitas antes dele.

O regime entra na LSTM como one-hot do estado mais provável segundo o filtro
Forward, com parâmetros do HMM estimados no treino: P(estado_t | obs_1..t)
usa apenas observações até t.

Alvo: log-retorno da carteira em t+1, previsto com informação disponível em t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.feature_selection import vif_filter
from src.models.hmm import fit_gaussian_hmm, forward_filter, select_n_states

# Variáveis de mercado: iguais para todos os ativos no painel, entram uma vez só.
MARKET_FEATURES = ("implied_vol_us", "implied_vol_br")
HMM_COLUMNS = ("return_1d", "realized_vol_21d", "realized_vol_63d")


@dataclass(frozen=True)
class WalkForwardSplit:
    fold: int
    train: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    test: pd.DatetimeIndex


def make_walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_years: int = 5,
    test_months: int = 6,
    step_months: int = 6,
    validation_fraction: float = 0.15,
) -> list[WalkForwardSplit]:
    """Janelas móveis: treino de `train_years` anos, teste dos `test_months` seguintes.

    A validação (para early stopping) é o trecho final do treino, em ordem
    cronológica — nunca uma amostra aleatória. A última janela pode ter teste
    mais curto quando a amostra acaba antes de completar `test_months`.
    """
    dates = pd.DatetimeIndex(dates).sort_values().unique()
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction deve estar entre 0 e 1.")

    splits = []
    start = dates[0]
    while True:
        test_start = start + pd.DateOffset(years=train_years)
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_start > dates[-1]:
            break
        window = dates[(dates >= start) & (dates < test_start)]
        test = dates[(dates >= test_start) & (dates < test_end)]
        if len(test) == 0:
            break
        n_val = max(1, int(round(len(window) * validation_fraction)))
        splits.append(WalkForwardSplit(len(splits), window[:-n_val], window[-n_val:], test))
        start = start + pd.DateOffset(months=step_months)
    return splits


def _stack_assets(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Empilha (data × ativo) para que cada linha seja uma observação de um ativo."""
    assets = [a for a in panel.columns.get_level_values(0).unique() if a != "market"]
    frames = [panel[a].reindex(columns=features) for a in assets]
    return pd.concat(frames, axis=0)


def select_split_features(
    panel: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    vif_threshold: float = 10.0,
    max_missing: float = 0.2,
) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    """Escolhe as colunas de entrada usando somente o treino.

    1. Descarta a coluna (ativo, feature) se faltar em mais de `max_missing` do
       treino — por exemplo, `implied_vol_br` em janelas que ficam quase todas
       antes de 2015, quando a série ainda não existia.
    2. Aplica o filtro VIF no painel empilhado (todos os ativos juntos), para
       que o mesmo conjunto de features valha para todos os ativos.
    """
    train = panel.loc[panel.index.intersection(train_dates)]
    missing = train.isna().mean()
    available = missing[missing <= max_missing].index

    asset_features = sorted({f for a, f in available if f not in MARKET_FEATURES})
    market = sorted({f for _, f in available if f in MARKET_FEATURES})
    stacked = _stack_assets(train, asset_features).dropna()
    kept, vif_report = vif_filter(stacked, threshold=vif_threshold) if len(asset_features) > 1 else (asset_features, pd.DataFrame())

    columns = [(a, f) for a, f in available if f in kept]
    columns += [("market", f) for f in market]

    report = pd.DataFrame({"missing_fraction_train": missing})
    report["selected"] = report.index.isin(columns)
    vif = vif_report["vif"] if "vif" in vif_report else pd.Series(dtype=float)
    report["vif"] = report.index.get_level_values(1).map(vif)
    return columns, report


def market_level_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Move as variáveis de mercado para um bloco único ('market', feature)."""
    asset_cols = [c for c in panel.columns if c[1] not in MARKET_FEATURES]
    out = panel[asset_cols].copy()
    for feature in MARKET_FEATURES:
        cols = [c for c in panel.columns if c[1] == feature]
        if cols:
            out[("market", feature)] = panel[cols[0]]
    return out


@dataclass(frozen=True)
class Scaler:
    mean: pd.Series
    std: pd.Series

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Scaler":
        std = df.std()
        if (std == 0).any() or std.isna().any():
            raise ValueError(f"Desvio-padrão nulo no treino: {list(std[(std == 0) | std.isna()].index)}")
        return cls(df.mean(), std)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df[self.mean.index] - self.mean) / self.std


def _contiguous_lengths(valid: pd.Series) -> list[int]:
    groups = valid.ne(valid.shift()).cumsum()
    return valid[valid].groupby(groups[valid]).size().tolist()


def causal_regime_features(
    portfolio: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    until: pd.Timestamp,
    state_range: tuple[int, ...] = (2, 3, 4),
    n_init: int = 10,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, int]:
    """Regime one-hot via filtro Forward, com HMM estimado só no treino.

    O número de estados também é escolhido por BIC dentro do treino. A
    filtragem roda do início da amostra até `until` (fim do teste) com os
    parâmetros congelados: o regime em t depende apenas de observações até t.
    Trechos separados por dados faltantes viram sequências independentes.
    """
    obs = portfolio.loc[:until, list(HMM_COLUMNS)]
    valid = obs.notna().all(axis=1)

    train_valid = valid.loc[obs.index.isin(train_dates)]
    train_obs = obs.loc[train_valid.index[train_valid]].to_numpy()
    train_lengths = _contiguous_lengths(train_valid)

    # Um número de estados que não converge neste treino sai da busca, em vez de
    # derrubar a janela inteira (comum com 4 estados em janelas mais calmas).
    candidates = []
    for k in state_range:
        try:
            candidates.append(select_n_states(train_obs, (k,), n_init=n_init, random_state=random_state, lengths=train_lengths))
        except RuntimeError:
            continue
    if not candidates:
        raise RuntimeError(f"HMM não convergiu para nenhum número de estados em {list(state_range)}.")
    n_states = int(pd.concat(candidates)["bic"].idxmin())
    model = fit_gaussian_hmm(train_obs, n_states, n_init=n_init, random_state=random_state, lengths=train_lengths)

    probs = forward_filter(model, obs.loc[valid].to_numpy(), lengths=_contiguous_lengths(valid))
    one_hot = np.eye(n_states)[probs.argmax(axis=1)]
    columns = pd.MultiIndex.from_tuples([("regime", f"state_{k}") for k in range(n_states)])
    return pd.DataFrame(one_hot, index=obs.index[valid], columns=columns).reindex(obs.index), n_states


def make_sequences(
    X: pd.DataFrame, target: pd.Series, sample_dates: pd.DatetimeIndex, lookback: int
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Janelas (lookback × features) terminando em t, com alvo em t+1.

    `X` e `target` precisam estar no mesmo índice de pregões. Uma amostra só
    entra se as `lookback` linhas da janela estiverem completas e o alvo de
    t+1 existir — sem imputação. A janela pode usar dias anteriores ao início
    do bloco (por exemplo, teste olhando o fim do treino), pois é passado.
    """
    if not X.index.equals(target.index):
        raise ValueError("X e target devem compartilhar o mesmo índice.")
    values = X.to_numpy(dtype=float)
    next_target = target.shift(-1).to_numpy(dtype=float)
    row_ok = np.isfinite(values).all(axis=1)
    complete_window = pd.Series(row_ok).rolling(lookback).sum().to_numpy() == lookback

    positions = X.index.get_indexer(sample_dates)
    positions = positions[positions >= 0]
    keep = positions[complete_window[positions] & np.isfinite(next_target[positions])]

    windows = np.stack([values[p - lookback + 1 : p + 1] for p in keep]) if len(keep) else np.empty((0, lookback, X.shape[1]))
    return windows, next_target[keep], X.index[keep]


def build_split_dataset(
    features_panel: pd.DataFrame,
    portfolio: pd.DataFrame,
    split: WalkForwardSplit,
    lookback: int = 42,
    vif_threshold: float = 10.0,
    max_missing: float = 0.2,
    state_range: tuple[int, ...] = (2, 3, 4),
    hmm_n_init: int = 10,
    random_state: int | None = None,
    hmm_observations: pd.DataFrame | None = None,
) -> dict:
    """Dataset pronto para a LSTM em uma janela walk-forward.

    Retorna arrays de treino/validação/teste, as colunas escolhidas, o número
    de estados do HMM e o desvio-padrão usado para escalar o alvo (para voltar
    à escala original de retorno na hora de calcular VaR/ES).

    hmm_observations permite aplicar, por exemplo, uma defasagem às entradas
    do HMM sem deslocar o alvo da carteira. Deve ter o mesmo índice e somente
    informações disponíveis na origem. O padrão preserva o comportamento
    anterior. O chamador é responsável pela disponibilidade dos dados.
    """
    panel = market_level_panel(features_panel)
    columns, selection_report = select_split_features(panel, split.train, vif_threshold, max_missing)

    until = split.test[-1]
    observations = portfolio if hmm_observations is None else hmm_observations
    if not observations.index.equals(portfolio.index):
        raise ValueError("Observações do HMM e carteira devem usar o mesmo calendário completo.")
    if not set(HMM_COLUMNS).issubset(observations.columns):
        raise ValueError("Observações do HMM sem as colunas necessárias.")
    regimes, n_states = causal_regime_features(observations, split.train, until, state_range, hmm_n_init, random_state)

    X = panel.loc[:until, columns]
    scaler = Scaler.fit(X.loc[X.index.intersection(split.train)])
    X = pd.concat([scaler.transform(X), regimes.reindex(X.index)], axis=1)

    target = portfolio["return_1d"].reindex(X.index)
    target_std = float(target.loc[target.index.intersection(split.train)].std())
    scaled_target = target / target_std

    out = {"fold": split.fold, "columns": list(X.columns), "n_states": n_states,
           "target_std": target_std, "selection_report": selection_report}
    for name, dates in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        # Último dia de cada bloco fica de fora: seu alvo (t+1) já pertence ao bloco seguinte.
        out[f"X_{name}"], out[f"y_{name}"], out[f"dates_{name}"] = make_sequences(X, scaled_target, dates[:-1], lookback)
    return out
