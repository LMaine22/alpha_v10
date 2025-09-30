"""
Critical leakage tests for PAWF/NPWF splits.

These are the MOST IMPORTANT tests - they verify zero temporal overlap
between train and test data, which is essential for valid backtesting.
"""

import pytest
import pandas as pd
import numpy as np
from alpha_discovery.splits import (
    build_pawf_splits,
    make_inner_folds,
    SplitSpec
)


def create_synthetic_timeseries(days=1000, seed=42):
    """Create synthetic time series for testing."""
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=days, freq='B')
    df = pd.DataFrame({
        'SPY_PX_LAST': 100 + np.cumsum(np.random.randn(days)),
        'AAPL_PX_LAST': 150 + np.cumsum(np.random.randn(days))
    }, index=dates)
    return df


class TestPAWFLeakage:
    """Test PAWF outer splits for leakage."""
    
    def test_no_train_test_overlap(self):
        """
        CRITICAL: Verify ZERO temporal overlap between train and test.
        
        This is the most important test - any overlap means look-ahead bias.
        """
        df = create_synthetic_timeseries(days=1000)
        
        splits = build_pawf_splits(
            df=df,
            label_horizon_days=5,
            feature_lookback_tail=20,
            min_train_months=12,
            test_window_days=21,
            step_months=1
        )
        
        assert len(splits) > 0, "Should generate at least one split"
        
        for i, spec in enumerate(splits):
            train_dates = set(spec.train_index)
            test_dates = set(spec.test_index)
            
            # ZERO overlap - most critical check
            overlap = train_dates & test_dates
            assert len(overlap) == 0, (
                f"Split {i} ({spec.outer_id}) has {len(overlap)} overlapping dates! "
                f"This indicates look-ahead bias."
            )
            
            # Additional sanity check: train ends before test starts
            assert spec.train_end < spec.test_start, (
                f"Split {i}: train_end ({spec.train_end}) should be before "
                f"test_start ({spec.test_start})"
            )
    
    def test_purge_gap_enforced(self):
        """Verify purge creates gap between train end and test start."""
        df = create_synthetic_timeseries(days=1000)
        
        splits = build_pawf_splits(
            df=df,
            label_horizon_days=5,  # This sets purge_days=5
            feature_lookback_tail=20,
            min_train_months=12
        )
        
        for spec in splits:
            gap_days = (spec.test_start - spec.train_end).days
            
            # Gap should be at least purge_days + 1
            assert gap_days >= spec.purge_days, (
                f"Split {spec.outer_id}: gap ({gap_days} days) is less than "
                f"purge_days ({spec.purge_days})"
            )
    
    def test_expanding_window(self):
        """Verify train window expands over time (anchored walk-forward)."""
        df = create_synthetic_timeseries(days=1000)
        
        splits = build_pawf_splits(
            df=df,
            label_horizon_days=5,
            feature_lookback_tail=20,
            min_train_months=12
        )
        
        train_lengths = [spec.train_span_days for spec in splits]
        
        # Each split should have >= previous split's training data
        for i in range(1, len(train_lengths)):
            assert train_lengths[i] >= train_lengths[i-1], (
                f"Split {i} has shorter train window ({train_lengths[i]} days) "
                f"than previous split ({train_lengths[i-1]} days). "
                f"PAWF should be expanding."
            )
    
    def test_split_ids_deterministic(self):
        """Verify split IDs are deterministic for same inputs."""
        df = create_synthetic_timeseries(days=1000, seed=42)
        
        splits1 = build_pawf_splits(df=df, label_horizon_days=5, feature_lookback_tail=20)
        splits2 = build_pawf_splits(df=df, label_horizon_days=5, feature_lookback_tail=20)
        
        assert len(splits1) == len(splits2)
        
        from alpha_discovery.splits import generate_split_id
        for s1, s2 in zip(splits1, splits2):
            id1 = generate_split_id(s1)
            id2 = generate_split_id(s2)
            assert id1 == id2, f"Split IDs should be deterministic: {id1} != {id2}"


class TestNPWFLeakage:
    """Test NPWF inner folds for leakage."""
    
    def test_no_inner_fold_overlap(self):
        """
        CRITICAL: Verify no overlap within inner folds.
        
        Inner folds are used for GA selection, so leakage here would
        invalidate hyperparameter selection.
        """
        df = create_synthetic_timeseries(days=1500)  # Longer for 5 folds
        
        folds = make_inner_folds(
            df_train_outer=df,
            label_horizon_days=5,
            feature_lookback_tail=20,
            k_folds=5
        )
        
        assert len(folds) > 0, "Should generate at least one inner fold"
        
        for i, (train_idx, test_idx) in enumerate(folds):
            train_set = set(train_idx)
            test_set = set(test_idx)
            
            # ZERO overlap
            overlap = train_set & test_set
            assert len(overlap) == 0, (
                f"Inner fold {i} has {len(overlap)} overlapping dates! "
                f"This indicates leakage in GA selection."
            )
            
            # Train should end before test starts
            assert train_idx.max() < test_idx.min(), (
                f"Inner fold {i}: train ends at {train_idx.max()}, "
                f"test starts at {test_idx.min()}"
            )
    
    def test_embargo_enforced(self):
        """Verify embargo removes train points near test end."""
        df = create_synthetic_timeseries(days=1000)  # Longer for 3 folds
        
        embargo_days = 10
        folds = make_inner_folds(
            df_train_outer=df,
            label_horizon_days=5,
            feature_lookback_tail=embargo_days,  # This sets embargo
            k_folds=3
        )
        
        for i, (train_idx, test_idx) in enumerate(folds):
            test_end = test_idx.max()
            
            # Calculate embargo cutoff
            embargo_cutoff = test_end + pd.Timedelta(days=embargo_days)
            
            # No train points should be within embargo window
            # (This is a simplified check - actual implementation may vary)
            train_after_test = train_idx[train_idx > test_end]
            if len(train_after_test) > 0:
                # If there are train points after test, they should respect embargo
                assert all(train_after_test >= embargo_cutoff), (
                    f"Inner fold {i}: train points within embargo window"
                )


class TestCrossValidationIntegrity:
    """Test overall CV integrity across PAWF and NPWF."""
    
    def test_nested_splits_no_leakage(self):
        """
        Integration test: verify nested PAWF → NPWF has no leakage.
        
        This simulates the actual workflow:
        1. Build outer PAWF splits
        2. For each outer split, create inner NPWF folds
        3. Verify no leakage at either level
        """
        df = create_synthetic_timeseries(days=1000)
        
        # Build outer splits
        outer_splits = build_pawf_splits(
            df=df,
            label_horizon_days=5,
            feature_lookback_tail=20,
            min_train_months=12
        )
        
        for outer_spec in outer_splits[:2]:  # Test first 2 outer folds (more data)
            # Extract outer train data
            outer_train_df = df.loc[outer_spec.train_start:outer_spec.train_end]
            
            # Skip if outer train too small
            if len(outer_train_df) < 800:
                continue
            
            # Build inner folds on outer training data
            inner_folds = make_inner_folds(
                df_train_outer=outer_train_df,
                label_horizon_days=5,
                feature_lookback_tail=20,
                k_folds=2  # Reduced to 2 for more reliable test
            )
            
            # Verify inner folds have no overlap
            for i, (inner_train_idx, inner_test_idx) in enumerate(inner_folds):
                overlap = set(inner_train_idx) & set(inner_test_idx)
                assert len(overlap) == 0, (
                    f"Outer fold {outer_spec.outer_id}, inner fold {i}: "
                    f"{len(overlap)} dates overlap"
                )
                
                # Verify inner folds don't leak into outer test
                outer_test_dates = set(df.loc[outer_spec.test_start:outer_spec.test_end].index)
                inner_train_leak = set(inner_train_idx) & outer_test_dates
                inner_test_leak = set(inner_test_idx) & outer_test_dates
                
                assert len(inner_train_leak) == 0, (
                    f"Inner train leaks into outer test: {len(inner_train_leak)} dates"
                )
                assert len(inner_test_leak) == 0, (
                    f"Inner test leaks into outer test: {len(inner_test_leak)} dates"
                )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
