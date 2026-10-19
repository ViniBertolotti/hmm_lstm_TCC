"""Quatro blocos cronológicos: ajuste, seleção, calibração e avaliação externa.

O horizonte é uma sessão. Os cinco anos são a janela de desenvolvimento:
reservar calibração dentro dela reduz o período efetivo de ajuste da LSTM.
Esta escolha deve constar do protocolo; não amplia silenciosamente a janela.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from src.data.dataset import WalkForwardSplit, build_split_dataset, make_walk_forward_splits


BLOCKS = ('train', 'validation', 'calibration', 'test')


def _dates(values, name):
    result = pd.DatetimeIndex(values)
    if result.empty or result.hasnans or result.has_duplicates or not result.is_monotonic_increasing:
        raise ValueError(f'{name}: datas devem ser crescentes, únicas, completas e não vazias.')
    return result


@dataclass(frozen=True)
class RiskWalkForwardSplit:
    fold: int
    train: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    calibration: pd.DatetimeIndex
    test: pd.DatetimeIndex


def _validate_split(split, sessions):
    previous = None
    blocks = {}
    for name in BLOCKS:
        dates = _dates(getattr(split, name), name)
        positions = sessions.get_indexer(dates)
        if len(dates) < 2 or (positions < 0).any() or (np.diff(positions) != 1).any():
            raise ValueError(f'{name}: exige pelo menos duas sessões consecutivas do calendário.')
        if previous is not None and previous >= dates[0]:
            raise ValueError('Os quatro blocos devem ser separados e ordenados no tempo.')
        previous = dates[-1]
        blocks[name] = dates
    return blocks


def make_risk_walk_forward_splits(sessions, *, development_years=5, test_months=6,
                                  step_months=6, calibration_sessions=252,
                                  validation_fraction=.15):
    """Reserva calibração no fim de cada janela de desenvolvimento.

    A fração de validação aplica-se ao que resta antes da calibração. Nenhum
    bloco depende de valores de retornos, regimes ou desempenho dos modelos.
    Amostras pequenas são recusadas, sem reduzir a calibração automaticamente.
    """
    calendar = _dates(sessions, 'sessions')
    for name, value in [('development_years', development_years), ('test_months', test_months),
                        ('step_months', step_months), ('calibration_sessions', calibration_sessions)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f'{name} deve ser inteiro positivo.')
    if calibration_sessions < 2 or not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError('Calibração exige >=2 sessões e fração de validação em (0,1).')
    base = make_walk_forward_splits(calendar, train_years=development_years,
        test_months=test_months, step_months=step_months, validation_fraction=validation_fraction)
    result = []
    for window in base:
        if len(window.test) < 2:  # janela final truncada pelo fim da amostra: sem teste avaliável
            continue
        development = window.train.append(window.validation)
        remaining = development[:-calibration_sessions]
        n_validation = max(2, round(len(remaining)*validation_fraction))
        if len(remaining)-n_validation < 2 or len(development) < calibration_sessions+4:
            raise ValueError('Janela insuficiente para os quatro blocos solicitados.')
        split = RiskWalkForwardSplit(window.fold, remaining[:-n_validation],
            remaining[-n_validation:], development[-calibration_sessions:], window.test)
        _validate_split(split, calendar)
        result.append(split)
    return result


def build_risk_split_dataset(features_panel, portfolio, split, *, sessions, **dataset_options):
    """Reutiliza seleção, scaler, HMM Forward e sequências do dataset existente.

    A união de seleção+calibração serve apenas para aplicar as transformações
    congeladas. Depois os arrays são separados novamente e qualquer alvo que
    atravesse uma fronteira é removido. A calibração nunca vai para fit_lstm.
    O calendário completo é obrigatório: lacunas de valores ficam como NaN.
    """
    calendar = _dates(sessions, 'sessions')
    if not features_panel.index.equals(calendar) or not portfolio.index.equals(calendar):
        raise ValueError('Painel e carteira devem conter todas as sessões, inclusive valores ausentes.')
    blocks = _validate_split(split, calendar)
    surrogate = WalkForwardSplit(split.fold, blocks['train'],
        blocks['validation'].append(blocks['calibration']), blocks['test'])
    data = build_split_dataset(features_panel, portfolio, surrogate, **dataset_options)
    if not np.isfinite(data['target_std']) or data['target_std'] <= 0:
        raise ValueError('Escala do alvo no treino deve ser positiva e finita.')
    source_origins = data['dates_validation']
    source_targets = calendar.take(calendar.get_indexer(source_origins)+1)
    source_x, source_y = data['X_validation'], data['y_validation']
    for name in ('validation', 'calibration'):
        mask = source_origins.isin(blocks[name]) & source_targets.isin(blocks[name])
        data[f'X_{name}'] = source_x[mask]
        data[f'y_{name}'] = source_y[mask]
        data[f'dates_{name}'] = source_origins[mask]
    for name in BLOCKS:
        origins = data[f'dates_{name}']
        targets = calendar.take(calendar.get_indexer(origins)+1)
        if origins.empty or not origins.isin(blocks[name]).all() or not targets.isin(blocks[name]).all():
            raise ValueError(f'{name}: bloco sem amostras válidas ou alvo atravessando sua fronteira.')
        data[f'target_dates_{name}'] = targets
    data['sessions'] = calendar
    data['model_fit_end'] = blocks['train'][-1]
    data['model_selection_end'] = blocks['validation'][-1]
    data['calibration_available_at'] = blocks['calibration'][-1]
    data['blocks'] = blocks
    return data
