"""LSTM mínima: ajuste em treino, early stopping em validação, teste externo.

Arrays seguem build_split_dataset: (amostras, lookback, features), alvo escalado.
Este módulo não estima VaR/ES a partir da dispersão das previsões.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import random

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class TrainingConfig:
    hidden: tuple[int, int] = (64, 32)
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 15
    seed: int = 42

    def __post_init__(self):
        if len(self.hidden) != 2 or any(h <= 0 for h in self.hidden):
            raise ValueError('hidden deve conter dois tamanhos positivos.')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout deve estar em [0, 1).')
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate deve ser positiva e finita.')
        if min(self.batch_size, self.max_epochs, self.patience) <= 0:
            raise ValueError('batch_size, max_epochs e patience devem ser positivos.')


class ReturnLSTM(nn.Module):
    def __init__(self, n_features: int, config: TrainingConfig):
        super().__init__()
        self.first = nn.LSTM(n_features, config.hidden[0], batch_first=True)
        self.dropout_sequence = nn.Dropout(config.dropout)
        self.second = nn.LSTM(config.hidden[0], config.hidden[1], batch_first=True)
        self.dropout_output = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.hidden[1], 1)

    def forward(self, x):
        x, _ = self.first(x)
        x, _ = self.second(self.dropout_sequence(x))
        return self.output(self.dropout_output(x[:, -1])).squeeze(-1)


def _arrays(X, y=None):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError('X deve ser 3D, não vazio e finito.')
    xt = torch.from_numpy(X.copy())
    if y is None:
        return xt
    y = np.asarray(y, dtype=np.float32)
    if y.shape != (len(X),) or not np.isfinite(y).all():
        raise ValueError('y deve ser finito e ter uma observação por sequência.')
    return xt, torch.from_numpy(y.copy())


def fit_lstm(X_train, y_train, X_validation, y_validation,
             config: TrainingConfig = TrainingConfig()):
    """Não recebe teste: escolha de época usa exclusivamente a validação.

    Execução CPU determinística. Mini-batches cronológicos; estados ocultos
    reiniciados em cada sequência. Retorna o checkpoint de menor MSE validado.
    """
    train_x, train_y = _arrays(X_train, y_train)
    val_x, val_y = _arrays(X_validation, y_validation)
    if train_x.shape[1:] != val_x.shape[1:]:
        raise ValueError('Treino e validação devem ter mesmo lookback e features.')
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model = ReturnLSTM(train_x.shape[2], config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    best_loss, best_epoch, stale, checkpoint = float('inf'), 0, 0, None
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total = 0.0
        for start in range(0, len(train_x), config.batch_size):
            x = train_x[start:start + config.batch_size]
            y = train_y[start:start + config.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError('Loss não finita durante treinamento.')
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
        model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(model(val_x), val_y).item()
        if not np.isfinite(validation_loss):
            raise RuntimeError('Loss não finita na validação.')
        history.append({'epoch': epoch, 'train_mse': total / len(train_x),
                        'validation_mse': validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            checkpoint = deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(checkpoint)
    model.eval()
    return model, {'config': asdict(config), 'best_epoch': best_epoch,
                   'best_validation_mse': best_loss, 'history': history,
                   'torch_version': torch.__version__, 'device': 'cpu'}


def predict(model: ReturnLSTM, X, target_std: float = 1.0):
    """Previsão pontual, dropout desligado, na escala original do alvo."""
    if not np.isfinite(target_std) or target_std <= 0:
        raise ValueError('target_std deve ser positivo e finito.')
    x = _arrays(X)
    model.eval()
    with torch.no_grad():
        result = model(x).numpy() * target_std
    if not np.isfinite(result).all():
        raise RuntimeError('Previsão não finita.')
    return result
