"""Dispersão por dropout estocástico; não é distribuição de perdas.

O dropout entre LSTMs usa máscaras independentes no tempo. Não se afirma
equivalência ao dropout variacional recorrente de Gal e Ghahramani (2016).
Quantis destes draws descrevem saídas da rede, não cobertura do retorno futuro.
"""
from numbers import Integral

import numpy as np
import torch
from torch import nn

from src.models.lstm import _arrays


def sample_predictions(model, X, n_samples=1000, seed=42, target_std=1.0):
    """Retorna (passagens, observações), sem alterar pesos, modos ou RNG externo.

    CPU; ativa somente módulos Dropout explícitos. BatchNorm e LSTMs ficam
    em avaliação. Modelo deve ter sido treinado com esses mesmos dropouts.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 2:
        raise ValueError('n_samples deve ser inteiro >= 2.')
    if not np.isfinite(target_std) or target_std <= 0:
        raise ValueError('target_std deve ser positivo e finito.')
    if any(parameter.device.type != 'cpu' for parameter in model.parameters()):
        raise ValueError('Esta implementação reproduzível aceita somente CPU.')
    x = _arrays(X)
    modules = list(model.modules())
    states = [module.training for module in modules]
    dropouts = [module for module in modules if isinstance(module, nn.Dropout)]
    if not any(module.p > 0 for module in dropouts):
        raise ValueError('Modelo não contém dropout ativo com p > 0.')
    try:
        model.eval()
        for module in dropouts:
            module.train()
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(seed)
            draws = np.stack([model(x).numpy().copy() for _ in range(n_samples)])
        draws *= target_std
        if not np.isfinite(draws).all():
            raise RuntimeError('Previsão MC não finita.')
        return draws
    finally:
        for module, state in zip(modules, states):
            module.training = state


def summarize_predictions(draws):
    """Resumo descritivo; sd/sqrt(M) é erro Monte Carlo da média das saídas."""
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2 or draws.shape[0] < 2 or draws.shape[1] == 0 or not np.isfinite(draws).all():
        raise ValueError('draws deve ser matriz finita (>=2 passagens, >=1 observação).')
    sd = draws.std(axis=0, ddof=1)
    return {'mean': draws.mean(axis=0), 'dropout_sd': sd,
            'mc_standard_error_mean': sd / np.sqrt(len(draws)),
            'output_q025': np.quantile(draws, .025, axis=0),
            'output_q975': np.quantile(draws, .975, axis=0)}


def sample_predictions_by_origin(model, X, origins, n_samples=1000, seed=42, target_std=1.0):
    """MC por origem, invariável à inclusão ou remoção de outras previsões.

    Uma semente derivada da data identifica cada previsão. As passagens MC
    viram linhas independentes do mesmo batch; o estado recorrente reinicia
    em cada linha. Isso é adequado a ReturnLSTM, sem BatchNorm em treino.
    É uma alternativa à amostragem conjunta para o executor walk-forward;
    seus draws não precisam ser idênticos aos da função sample_predictions.
    """
    import hashlib
    import pandas as pd

    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 2:
        raise ValueError('n_samples deve ser inteiro >= 2.')
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError('seed deve ser inteiro.')
    if not np.isfinite(target_std) or target_std <= 0:
        raise ValueError('target_std deve ser positivo e finito.')
    x = _arrays(X)
    dates = pd.DatetimeIndex(origins)
    if len(dates) != len(x) or dates.hasnans or dates.has_duplicates:
        raise ValueError('Exige uma origem única e completa por sequência.')
    if any(parameter.device.type != 'cpu' for parameter in model.parameters()):
        raise ValueError('Esta implementação reproduzível aceita somente CPU.')
    modules = list(model.modules())
    modes = [module.training for module in modules]
    dropouts = [module for module in modules if isinstance(module, nn.Dropout)]
    if not any(module.p > 0 for module in dropouts):
        raise ValueError('Modelo não contém dropout ativo com p > 0.')
    draws = np.empty((n_samples, len(x)), dtype=np.float32)
    try:
        model.eval()
        for module in dropouts:
            module.train()
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            for position, origin in enumerate(dates):
                identity = f'{seed}:{origin.value}'.encode('utf-8')
                origin_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], 'little') % (2**63-1)
                torch.manual_seed(origin_seed)
                repeated = x[position:position+1].expand(n_samples, -1, -1)
                output = model(repeated).numpy()
                if output.shape != (n_samples,):
                    raise ValueError('Modelo deve retornar uma previsão escalar por sequência.')
                draws[:, position] = output * target_std
        if not np.isfinite(draws).all():
            raise RuntimeError('Previsão MC não finita.')
        return draws
    finally:
        for module, mode in zip(modules, modes):
            module.training = mode
