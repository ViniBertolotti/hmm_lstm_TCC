"""Encadeamento auditável por razão com cronograma explícito de contratos.

Convenção: troca ao fechamento de t-1; t é a primeira sessão com OHLC do novo
contrato. Fator = Close_novo(t-1) / Close_antigo(t-1). Assim, o retorno em t
é Close_novo(t) / Close_novo(t-1) - 1, preservando o movimento de mercado.
Não calcula P&L de contratos, custos, margem, colateral ou rendimento do caixa.
"""
import numpy as np
import pandas as pd

from src.data.collectors import adjust_ohlc_roll as _adjust_ohlc_roll

PRICE_COLUMNS = ['Open', 'High', 'Low', 'Close']


def build_roll_audit(quotes, schedule, metadata, sessions, initial_contract,
                     adjust_ohlc_roll=_adjust_ohlc_roll):
    """Retorna série bruta, back-adjusted, índice causal e tabela de fatores.

    quotes: DataFrame indexado por (date, contract), com OHLCV.
    schedule: roll_date, old_contract, new_contract, decision_date.
    metadata: índice contract, first_notice_date e last_trade_date reais.
    sessions: calendário completo da bolsa do futuro, não calendário B3.
    decision_date: data em que o cronograma estava definido/conhecido.
    adjust_ohlc_roll: função existente do projeto, reutilizada sem alteração.

    Os metadados e a regra de decisão precisam de comprovação externa. Esta
    função valida consistência dos dados fornecidos, não sua autenticidade.
    """
    sessions = pd.DatetimeIndex(sessions)
    if len(sessions) < 2 or sessions.hasnans or sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError('Calendário deve ser crescente, único, sem NaT e ter >=2 sessões.')
    if not isinstance(quotes.index, pd.MultiIndex) or quotes.index.names != ['date', 'contract']:
        raise ValueError('quotes deve ter índice (date, contract).')
    if quotes.index.has_duplicates:
        raise ValueError('Cotação duplicada por data e contrato.')
    if not set(PRICE_COLUMNS + ['Volume']).issubset(quotes.columns):
        raise ValueError('OHLCV incompleto.')
    if metadata.index.has_duplicates:
        raise ValueError('Metadados duplicados por contrato.')
    required = ['roll_date', 'old_contract', 'new_contract', 'decision_date']
    if not set(required).issubset(schedule.columns):
        raise ValueError('Cronograma incompleto.')
    schedule = schedule.copy()
    for column in ['roll_date', 'decision_date']:
        schedule[column] = pd.to_datetime(schedule[column], errors='raise')
        if schedule[column].isna().any():
            raise ValueError('Data ausente no cronograma.')
    if schedule.roll_date.duplicated().any() or not schedule.roll_date.is_monotonic_increasing:
        raise ValueError('Cronograma deve ser crescente e sem rolagens duplicadas.')

    def limits(contract):
        if contract not in metadata.index:
            raise ValueError(f'Metadados ausentes: {contract}')
        fnd = pd.Timestamp(metadata.loc[contract, 'first_notice_date'])
        ltd = pd.Timestamp(metadata.loc[contract, 'last_trade_date'])
        if pd.isna(fnd) or pd.isna(ltd):
            raise ValueError(f'Datas contratuais ausentes: {contract}')
        return fnd, ltd

    def close(date, contract):
        try:
            value = float(quotes.loc[(date, contract), 'Close'])
        except KeyError as exc:
            raise ValueError(f'Falta cotação simultânea: {date}, {contract}') from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Preços do fator devem ser positivos e finitos.')
        return value

    active = pd.Series(initial_contract, index=sessions, dtype=object)
    current = initial_contract
    records = []
    for row in schedule.itertuples(index=False):
        position = sessions.get_indexer([row.roll_date])[0]
        if position <= 0:
            raise ValueError('Rolagem deve ser sessão do calendário com sessão anterior.')
        reference = sessions[position - 1]
        if row.decision_date > reference:
            raise ValueError('Decisão de rolagem posterior à data de execução.')
        if row.old_contract != current or row.new_contract == current:
            raise ValueError('Cadeia de contratos inconsistente.')
        for contract in (current, row.new_contract):
            fnd, ltd = limits(contract)
            if reference >= min(fnd, ltd):
                raise ValueError('Troca precisa ocorrer antes do FND e último pregão.')
        old = close(reference, current)
        new = close(reference, row.new_contract)
        factor = new / old
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError('Fator inválido.')
        records.append({'roll_date': row.roll_date, 'reference_date': reference,
            'decision_date': row.decision_date, 'old_contract': current,
            'new_contract': row.new_contract, 'old_close': old,
            'new_close': new, 'factor': factor})
        active.loc[row.roll_date:] = row.new_contract
        current = row.new_contract

    keys = pd.MultiIndex.from_arrays([sessions, active], names=['date', 'contract'])
    raw = quotes.reindex(keys)[PRICE_COLUMNS + ['Volume']].copy()
    raw.index = sessions
    prices = raw[PRICE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('OHLC ativo ausente ou inválido; sem preenchimento automático.')
    if ((raw.High < raw[['Open', 'Close', 'Low']].max(axis=1)) |
        (raw.Low > raw[['Open', 'Close', 'High']].min(axis=1))).any():
        raise ValueError('Geometria OHLC inconsistente.')
    volume = raw.Volume.to_numpy(dtype=float)
    if np.isinf(volume).any() or (volume < 0).any():
        raise ValueError('Volume negativo ou infinito.')
    for date, contract in active.items():
        if date >= min(limits(contract)):
            raise ValueError(f'Contrato ativo alcança FND/último pregão: {contract}, {date}')
    ledger = pd.DataFrame(records, columns=['roll_date', 'reference_date', 'decision_date',
        'old_contract', 'new_contract', 'old_close', 'new_close', 'factor'])
    factors = ledger.set_index('roll_date').factor
    adjusted = adjust_ohlc_roll(raw, roll_factors=factors)
    # Encadeamento progressivo não muda o passado quando novas rolagens chegam.
    gross = raw.Close / raw.Close.shift(1)
    for record in records:
        gross.loc[record['roll_date']] /= record['factor']
    causal_index = 100 * gross.fillna(1).cumprod()
    causal_index.name = 'causal_return_index'
    pd.testing.assert_series_equal(adjusted.Volume, raw.Volume)
    np.testing.assert_allclose(adjusted.Close.pct_change().iloc[1:], gross.iloc[1:] - 1,
                               atol=1e-12)
    return {'raw': raw, 'adjusted': adjusted, 'active_contract': active,
            'ledger': ledger, 'causal_index': causal_index,
            'returns': gross - 1}
