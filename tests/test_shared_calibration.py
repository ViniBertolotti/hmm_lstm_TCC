import numpy as np
import pandas as pd

from scripts.build_shared_calibration import shared_predictions


def test_ablation_changes_only_hmm_risk_even_with_duplicate_saved_indices():
    frame = pd.DataFrame({'model': ['lstm', 'hmm_lstm', 'hmm_lstm'],
        'prediction': [0., 0., 0.], 'scale': [1., 1., 1.], 'regime': [np.nan, 0, 1],
        'confidence': [.95, .95, .95], 'var': [123., 456., 789.],
        'es': [124., 457., 790.], 'residual_pool': ['all', 'regime_0', 'regime_1']},
        index=[0, 0, 1])
    cal = pd.DataFrame({'model': ['hmm_lstm'] * 100, 'prediction': 0.,
        'return_1d': np.linspace(-.03, .03, 100), 'scale': 1.})
    before = frame.copy(deep=True)
    result = shared_predictions(frame, cal)
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(result.query("model == 'lstm'").reset_index(drop=True),
                                  frame.query("model == 'lstm'").reset_index(drop=True))
    hmm = result.query("model == 'hmm_lstm'")
    assert hmm['var'].nunique() == hmm.es.nunique() == 1
    assert set(hmm.residual_pool) == {'all'}
    assert list(hmm.regime) == [0, 1]
