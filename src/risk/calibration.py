"""Distribuição empírica de erros posteriores à seleção do modelo.

    z_i=(r_i-mu_i)/s_i; R(t+1)=mu(t)+s(t)*Z_empírico.

A média MC pode ser mu. A dispersão MC permanece diagnóstico separado: somá-la
à distribuição de erros já observados pode contar incerteza duas vezes.
Escalas s são opcionais (default 1), positivas e devem ser conhecidas na origem.
Não há garantia de cobertura: estabilidade dos erros deve ser avaliada fora da
amostra, especialmente com heteroscedasticidade e mudança de regimes.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from src.risk.var_es import empirical_var_es, losses_from_log_returns


def _timestamp(value):
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("Timestamp ausente.")
    return result


def _sessions(values):
    result = pd.DatetimeIndex(values)
    if len(result) < 3 or result.has_duplicates or result.hasnans or not result.is_monotonic_increasing:
        raise ValueError("Calendário deve ser crescente, único e sem NaT.")
    return result


def _check_next_session(calendar, origins, targets):
    origins, targets = pd.DatetimeIndex(origins), pd.DatetimeIndex(targets)
    positions = calendar.get_indexer(origins)
    if (positions < 0).any() or (positions >= len(calendar)-1).any():
        raise ValueError("Origem fora do calendário ou sem sessão posterior.")
    if not calendar.take(positions + 1).equals(targets):
        raise ValueError("Realização deve corresponder à sessão imediatamente seguinte.")


@dataclass(frozen=True)
class ResidualCalibrator:
    residuals: tuple[float, ...]
    sessions: tuple[pd.Timestamp, ...]
    model_id: str
    model_fit_end: pd.Timestamp
    model_selection_end: pd.Timestamp
    fitted_as_of: pd.Timestamp
    calibration_first_origin: pd.Timestamp
    calibration_last_realization: pd.Timestamp

    def forecast(self, prediction, *, origin, target, model_id, scale=1.0,
                 confidence_levels=(.95, .99), horizon=1):
        """Risco de uma sessão, condicionado ao modelo congelado da calibração."""
        if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon != 1:
            raise ValueError("Somente horizonte de uma sessão foi implementado.")
        origin, target = _timestamp(origin), _timestamp(target)
        if origin < self.fitted_as_of or origin < self.calibration_last_realization:
            raise ValueError("Calibração usa informação posterior à origem pedida.")
        if model_id != self.model_id:
            raise ValueError("Modelo diferente exige nova calibração compatível.")
        if not np.isscalar(prediction) or not np.isfinite(prediction):
            raise ValueError("Previsão deve ser escalar finito em log-retorno.")
        if not np.isscalar(scale) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Escala da previsão deve ser positiva e finita.")
        _check_next_session(_sessions(self.sessions), [origin], [target])
        levels = tuple(confidence_levels)
        if not levels or len(set(levels)) != len(levels):
            raise ValueError("Informe níveis de confiança únicos e não vazios.")
        scenarios = prediction + scale * np.asarray(self.residuals)
        losses = losses_from_log_returns(scenarios)
        rows = [empirical_var_es(losses, level) for level in sorted(levels)]
        result = pd.DataFrame(rows).set_index("confidence")
        result["origin"] = origin
        result["target"] = target
        result["loss_definition"] = "negative_simple_return"
        result["horizon_sessions"] = 1
        return result


def fit_residual_calibrator(calibration, *, sessions, model_id,
                            model_fit_end, model_selection_end, as_of):
    """Exige bloco separado de treino e de early stopping/seleção.

    Colunas: origin, realized_at, prediction, return_1d e scale opcional.
    Metadados temporais são verificados, mas a função não autentica o histórico
    de treinamento nem comprova que previsões/escalas foram geradas causalmente.
    Essa evidência deve vir do executor do experimento e de seus checkpoints.
    """
    required = {"origin", "realized_at", "prediction", "return_1d"}
    if not required.issubset(calibration.columns) or len(calibration) < 2:
        raise ValueError("Bloco de calibração incompleto ou com menos de duas observações.")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("Identificador do modelo é obrigatório.")
    frame = calibration.copy()
    calendar = _sessions(sessions)
    origins = pd.DatetimeIndex(pd.to_datetime(frame.origin))
    realized = pd.DatetimeIndex(pd.to_datetime(frame.realized_at))
    if (origins.hasnans or realized.hasnans or origins.has_duplicates or realized.has_duplicates
            or not origins.is_monotonic_increasing or not realized.is_monotonic_increasing):
        raise ValueError("Previsões de calibração devem ter datas crescentes, únicas e completas.")
    _check_next_session(calendar, origins, realized)
    fit_end = _timestamp(model_fit_end)
    selection_end = _timestamp(model_selection_end)
    available = _timestamp(as_of)
    if fit_end > selection_end or selection_end > origins.min():
        raise ValueError("Treino e seleção devem terminar antes das previsões de calibração.")
    if realized.max() > available:
        raise ValueError("Há retornos de calibração ainda não observados em as_of.")
    numeric = frame[["prediction", "return_1d"]].to_numpy(dtype=float)
    scales = frame.scale.to_numpy(dtype=float) if "scale" in frame else np.ones(len(frame))
    if not np.isfinite(numeric).all() or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Preços previstos/observados e escalas devem ser finitos; escala positiva.")
    residuals = (numeric[:, 1] - numeric[:, 0]) / scales
    return ResidualCalibrator(tuple(map(float, residuals)), tuple(calendar), model_id,
                              fit_end, selection_end, available, origins.min(), realized.max())
