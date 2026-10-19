"""Executa uma janela já preparada, sem consultar teste para ajustar o modelo.

Calibração empírica congelada; nenhuma garantia de cobertura ou estabilidade
dos erros. O tamanho efetivo da cauda é registrado, inclusive a 99%.
O benchmark de média zero é apenas uma referência simples, não o MS-GARCH.
"""
import hashlib

import numpy as np
import pandas as pd

from src.models.lstm import TrainingConfig, fit_lstm
from src.models.mc_dropout import sample_predictions_by_origin, summarize_predictions
from src.risk.calibration import fit_residual_calibrator
from src.risk.var_es import losses_from_log_returns
from src.risk.backtests import (
    acerbi_szekely_unconditional, diebold_mariano, quantile_score, var_coverage,
)


def checkpoint_id(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode('utf-8'))
        array = value.detach().cpu().numpy()
        digest.update(str(array.shape).encode('ascii'))
        digest.update(str(array.dtype).encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_risk_fold(data, *, training_config=TrainingConfig(), mc_samples=1000,
                  mc_seed=42, include_regimes=True, confidence_levels=(.95, .99),
                  dm_max_lag=5):
    """Treina só em train/validation; calibra depois; pontua teste ao final.

    O chamador deve comprovar origem/publicação dos dados e transformações
    causais. Use build_risk_split_dataset; não transfira calibração para treino.
    Retorna artefatos em memória. Só o executor decide onde salvá-los.
    """
    levels = tuple(confidence_levels)
    if (not levels or len(set(levels)) != len(levels)
            or any(not np.isfinite(level) or not 0 < level < 1 for level in levels)):
        raise ValueError('Níveis de confiança devem ser únicos e estar em (0,1).')
    columns = data['columns']
    chosen = [i for i, column in enumerate(columns) if include_regimes or column[0] != 'regime']
    if not chosen:
        raise ValueError('Nenhuma variável de entrada selecionada.')
    blocks = ('train', 'validation', 'calibration', 'test')
    features = {name: data[f'X_{name}'][:, :, chosen] for name in blocks}
    model, training = fit_lstm(features['train'], data['y_train'],
        features['validation'], data['y_validation'], config=training_config)
    model_id = checkpoint_id(model)
    name = 'hmm_lstm' if include_regimes else 'lstm'
    summaries = {}
    for block in ('calibration', 'test'):
        draws = sample_predictions_by_origin(model, features[block], data[f'dates_{block}'],
            n_samples=mc_samples, seed=mc_seed, target_std=data['target_std'])
        summaries[block] = summarize_predictions(draws)
    calibration = pd.DataFrame({'origin': data['dates_calibration'],
        'realized_at': data['target_dates_calibration'],
        'prediction': summaries['calibration']['mean'],
        'return_1d': data['y_calibration']*data['target_std']})
    calibration['dropout_sd'] = summaries['calibration']['dropout_sd']
    metadata = dict(sessions=data['sessions'], model_fit_end=data['model_fit_end'],
        model_selection_end=data['model_selection_end'], as_of=data['calibration_available_at'])
    learned = fit_residual_calibrator(calibration, model_id=model_id, **metadata)
    zero_frame = calibration.copy(); zero_frame['prediction'] = 0.
    historical = fit_residual_calibrator(zero_frame, model_id='zero_mean_reference', **metadata)
    rows = []
    # No test target is passed to either forecasting function.
    for position, (origin, target) in enumerate(zip(data['dates_test'], data['target_dates_test'])):
        for label, calibrator, identity, prediction in (
                (name, learned, model_id, summaries['test']['mean'][position]),
                ('zero_mean_reference', historical, 'zero_mean_reference', 0.)):
            risks = calibrator.forecast(float(prediction), origin=origin, target=target,
                model_id=identity, confidence_levels=levels)
            for confidence, risk in risks.iterrows():
                rows.append({'model': label, 'origin': origin, 'target': target,
                    'confidence': float(confidence), 'prediction': float(prediction),
                    'dropout_sd': float(summaries['test']['dropout_sd'][position]) if label == name else 0.,
                    'var': float(risk['var']), 'es': float(risk['es']),
                    'calibration_n': len(calibration)})
    forecasts = pd.DataFrame(rows)
    actual = pd.Series(data['y_test']*data['target_std'], index=data['target_dates_test'])
    realized_losses = pd.Series(losses_from_log_returns(actual.to_numpy()), index=actual.index)
    forecasts['actual_log_return'] = forecasts.target.map(actual)
    forecasts['actual_loss'] = forecasts.target.map(realized_losses)
    report = {'fold': data['fold'], 'model': name, 'model_id': model_id,
        'training': training, 'selected_columns': [columns[i] for i in chosen],
        'n_states': data['n_states'] if include_regimes else None,
        'mc_samples': mc_samples, 'mc_seed': mc_seed,
        'mc_sampling': 'date_keyed_vectorized_draws_per_origin',
        'risk_method': 'frozen_empirical_errors_after_model_selection_scale_one',
        'loss': 'negative_simple_return', 'horizon_sessions': 1,
        'n_sequences': {block: len(data[f'y_{block}']) for block in blocks},
        'block_dates': {block: {'first': str(data['blocks'][block][0]),
                               'last': str(data['blocks'][block][-1])} for block in blocks},
        'effective_calibration_tail': {str(level): len(calibration)*(1-level) for level in levels},
        'point_mse': {}, 'coverage': {}, 'es_statistics_without_null_pvalue': {},
        'quantile_score_comparisons': {},
        'limitations': [
            'Financial source and availability must be established outside this function.',
            'Empirical residual stability and predictive risk coverage remain unproven.',
            'A small empirical tail, especially 99%, limits precision.',
            'Dropout spread is diagnostic and is not added again to empirical errors.',
            'ES reference paths were not generated; no A-S p-value is available.',
            'The zero-mean reference does not replace the deferred MSGARCH benchmark.',
        ]}
    for label in (name, 'zero_mean_reference'):
        one = forecasts[(forecasts.model == label) & (forecasts.confidence == levels[0])].set_index('target')
        report['point_mse'][label] = float(np.mean((one.prediction-actual)**2))
        report['coverage'][label] = {}; report['es_statistics_without_null_pvalue'][label] = {}
        for level in levels:
            subset = forecasts[(forecasts.model == label) & (forecasts.confidence == level)].set_index('target')
            report['coverage'][label][str(level)] = var_coverage(realized_losses, subset['var'],
                level, sessions=data['sessions'])
            if (subset.es > 0).all():
                es_test = acerbi_szekely_unconditional(realized_losses.to_numpy(),
                    subset['var'].to_numpy(), subset.es.to_numpy(), level)
            else:
                es_test = {'status': 'not_applicable_es_not_positive', 'p_value_lower_tail': None}
            report['es_statistics_without_null_pvalue'][label][str(level)] = es_test
    for level in levels:
        pair = []
        for label in (name, 'zero_mean_reference'):
            subset = forecasts[(forecasts.model == label) & (forecasts.confidence == level)].set_index('target')
            pair.append(pd.Series(quantile_score(realized_losses.to_numpy(), subset['var'].to_numpy(), level), index=actual.index))
        if len(actual) >= 3 and dm_max_lag < len(actual):
            report['quantile_score_comparisons'][str(level)] = diebold_mariano(*pair,
                sessions=data['sessions'], max_lag=dm_max_lag)
        else:
            report['quantile_score_comparisons'][str(level)] = {'status': 'insufficient_paired_observations'}
    if checkpoint_id(model) != model_id:
        raise RuntimeError('Checkpoint mudou durante calibração ou avaliação.')
    return {'model': model, 'manifest': report, 'calibration': calibration, 'forecasts': forecasts}
