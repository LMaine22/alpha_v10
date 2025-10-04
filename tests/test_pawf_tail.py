import pandas as pd

from alpha_discovery.splits.pawf import build_pawf_splits


def test_pawf_tail_reaches_data_end_and_adjusts_buffers():
    idx = pd.bdate_range('2022-01-03', periods=420)
    df = pd.DataFrame({'close': range(len(idx))}, index=idx)

    splits = build_pawf_splits(
        df=df,
        label_horizon_days=5,
        feature_lookback_tail=252,
        min_train_months=12,
        test_window_days=252,
        step_months=3,
    )

    assert splits, "Should create at least one split"
    assert splits[-1].test_end == idx[-1]

    tail_splits = [s for s in splits if getattr(s, 'is_tail', False)]
    assert len(tail_splits) == 1
    tail_notes = tail_splits[0].notes or {}
    adjustments = tail_notes.get('tail_buffer_adjustments', {})
    # Embargo must be adjusted because the tail window is shorter than the feature tail
    assert 'embargo_days' in adjustments
    assert adjustments['embargo_days']['from'] > adjustments['embargo_days']['to']
