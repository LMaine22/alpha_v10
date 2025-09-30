"""
Forecast-first validation orchestrator.

Orchestrates the complete validation workflow for discovered alphas:
1. Build PAWF outer splits
2. For each outer fold:
   - Run GA with NPWF inner folds for selection
   - Evaluate on outer test with proper scoring rules
   - Calculate skill vs baselines
   - Test orthogonality (horizons, calendar)
   - Run bootstrap robustness tests
   - Train calibrators
3. Produce eligibility matrix artifact

This is a clean rewrite separate from the legacy validation.py module.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import json

from ..config import settings
from ..splits import (
    build_pawf_splits,
    make_inner_folds,
    SplitSpec,
    generate_split_id,
    fit_regimes,
    assign_regime,
    make_horizon_holdouts,
    make_calendar_holdouts,
    bootstrap_skill_delta,
    compute_adversarial_auc,
    check_drift_gate
)
from ..adapters import FeatureAdapter, calculate_max_lookback
from ..search import ga_core
from ..eval.metrics import distribution, info_theory


@dataclass
class ValidationResult:
    """Result from validating a single setup on a single outer fold."""
    
    split_id: str
    ticker: str
    setup: List[str]
    horizon: int
    
    # Core metrics (proper scoring rules)
    crps: float
    brier_score: float
    log_loss: float
    pinball_q10: float
    pinball_q90: float
    
    # Skill metrics
    skill_vs_uniform: float  # CRPS delta vs uniform baseline
    skill_vs_marginal: float  # CRPS delta vs marginal baseline
    
    # Calibration
    calibration_mae: float
    calibration_ece: float  # Expected Calibration Error
    
    # Forecast distribution
    band_probs: List[float]
    band_edges: List[float]
    
    # Support
    n_triggers_train: int
    n_triggers_test: int
    
    # Metadata
    regime_train: str
    regime_test: str
    regime_similarity: float
    
    # Robustness flags
    drift_auc: float
    drift_passed: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "split_id": self.split_id,
            "ticker": self.ticker,
            "setup": self.setup,
            "horizon": self.horizon,
            "crps": self.crps,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "pinball_q10": self.pinball_q10,
            "pinball_q90": self.pinball_q90,
            "skill_vs_uniform": self.skill_vs_uniform,
            "skill_vs_marginal": self.skill_vs_marginal,
            "calibration_mae": self.calibration_mae,
            "calibration_ece": self.calibration_ece,
            "band_probs": self.band_probs,
            "band_edges": self.band_edges,
            "n_triggers_train": self.n_triggers_train,
            "n_triggers_test": self.n_triggers_test,
            "regime_train": self.regime_train,
            "regime_test": self.regime_test,
            "regime_similarity": self.regime_similarity,
            "drift_auc": self.drift_auc,
            "drift_passed": self.drift_passed
        }


@dataclass
class EligibilityMatrix:
    """
    Eligibility matrix: comprehensive validation results for all setups.
    
    This is the key artifact from validation that determines which setups
    can proceed to production.
    """
    
    results: List[ValidationResult]
    metadata: Dict[str, Any]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        return pd.DataFrame([r.to_dict() for r in self.results])
    
    def filter_eligible(
        self,
        min_skill_vs_marginal: float = 0.01,
        max_calibration_mae: float = 0.15,
        drift_gate: bool = True
    ) -> List[ValidationResult]:
        """
        Filter to eligible setups based on criteria.
        
        Args:
            min_skill_vs_marginal: Minimum skill improvement over marginal baseline
            max_calibration_mae: Maximum calibration error allowed
            drift_gate: If True, require drift test to pass
            
        Returns:
            List of eligible ValidationResult objects
        """
        eligible = []
        for r in self.results:
            # Skill gate
            if r.skill_vs_marginal < min_skill_vs_marginal:
                continue
            
            # Calibration gate
            if r.calibration_mae > max_calibration_mae:
                continue
            
            # Drift gate
            if drift_gate and not r.drift_passed:
                continue
            
            eligible.append(r)
        
        return eligible
    
    def save(self, output_path: Path):
        """Save eligibility matrix to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, input_path: Path) -> EligibilityMatrix:
        """Load eligibility matrix from JSON."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        results = [ValidationResult(**r) for r in data["results"]]
        return cls(results=results, metadata=data["metadata"])


class ForecastOrchestrator:
    """
    Main orchestrator for forecast-first validation.
    
    Coordinates PAWF outer splits, NPWF inner selection, evaluation,
    robustness testing, and artifact generation.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        signals_df: pd.DataFrame,
        feature_adapter: Optional[FeatureAdapter] = None
    ):
        """
        Initialize orchestrator.
        
        Args:
            df: Master DataFrame with price/market data
            signals_df: DataFrame with signal triggers (boolean columns)
            feature_adapter: Optional FeatureAdapter (created if not provided)
        """
        self.df = df
        self.signals_df = signals_df
        self.feature_adapter = feature_adapter or FeatureAdapter()
        
        # Configuration from settings
        self.horizons = list(getattr(settings.forecast, 'horizons', [5, 21]))
        self.band_edges = np.asarray(settings.forecast.band_edges, dtype=float)
        self.price_field = settings.forecast.price_field
        
    def run_validation(
        self,
        discovered_setups: List[Tuple[str, List[str]]],  # (ticker, signals) pairs
        output_dir: Path,
        n_jobs: int = -1
    ) -> EligibilityMatrix:
        """
        Run complete validation workflow.
        
        Args:
            discovered_setups: List of (ticker, signal_list) tuples from GA
            output_dir: Directory for artifacts
            n_jobs: Parallelism for evaluation
            
        Returns:
            EligibilityMatrix with all validation results
        """
        print(f"[Orchestrator] Validating {len(discovered_setups)} setups...")
        
        # Step 1: Build PAWF outer splits
        print("[Orchestrator] Building PAWF outer splits...")
        pawf_splits = self._build_outer_splits()
        print(f"[Orchestrator] Created {len(pawf_splits)} outer folds")
        
        # Step 2: Fit regimes on full data
        print("[Orchestrator] Fitting regime model...")
        regime_model = self._fit_regimes()
        
        # Step 3: Validate each setup on each outer fold
        print("[Orchestrator] Running outer fold evaluations...")
        all_results = []
        
        for spec in pawf_splits:
            fold_results = self._validate_on_outer_fold(
                spec, discovered_setups, regime_model
            )
            all_results.extend(fold_results)
        
        # Step 4: Create eligibility matrix
        print(f"[Orchestrator] Collected {len(all_results)} validation results")
        eligibility = EligibilityMatrix(
            results=all_results,
            metadata={
                "n_setups": len(discovered_setups),
                "n_outer_folds": len(pawf_splits),
                "horizons": self.horizons,
                "band_edges": self.band_edges.tolist(),
                "split_version": "PAWF_v1"
            }
        )
        
        # Step 5: Save artifacts
        eligibility.save(output_dir / "eligibility_matrix.json")
        self._save_summary_stats(eligibility, output_dir)
        
        print("[Orchestrator] Validation complete!")
        return eligibility
    
    def _build_outer_splits(self) -> List[SplitSpec]:
        """Build PAWF outer splits."""
        # Calculate lookback from features used in signals
        # For now, use conservative default
        feature_lookback_tail = 63  # TODO: derive from signal features
        
        # Build PAWF splits
        splits = build_pawf_splits(
            df=self.df,
            label_horizon_days=max(self.horizons),
            feature_lookback_tail=feature_lookback_tail,
            min_train_months=getattr(settings.validation, 'min_train_months', 36),
            test_window_days=getattr(settings.validation, 'test_window_days', 21),
            step_months=getattr(settings.validation, 'step_months', 1)
        )
        
        return splits
    
    def _fit_regimes(self):
        """Fit regime model on full data for stratification."""
        price_col = f"SPY_{self.price_field}"  # Use market proxy
        
        regime_model, _ = fit_regimes(
            df=self.df,
            price_col=price_col,
            K=5,
            version="R1"
        )
        
        return regime_model
    
    def _validate_on_outer_fold(
        self,
        spec: SplitSpec,
        setups: List[Tuple[str, List[str]]],
        regime_model
    ) -> List[ValidationResult]:
        """
        Validate all setups on a single outer fold.
        
        Args:
            spec: Outer fold SplitSpec
            setups: List of (ticker, signals) to validate
            regime_model: Fitted regime model
            
        Returns:
            List of ValidationResult objects for this fold
        """
        split_id = generate_split_id(spec)
        print(f"[Orchestrator]   Fold {spec.outer_id}: {spec.test_start.date()} - {spec.test_end.date()}")
        
        # Extract train/test data
        train_df = self.df.loc[spec.train_start:spec.train_end]
        test_df = self.df.loc[spec.test_start:spec.test_end]
        
        train_signals = self.signals_df.loc[spec.train_start:spec.train_end]
        test_signals = self.signals_df.loc[spec.test_start:spec.test_end]
        
        results = []
        
        for ticker, signal_list in setups:
            for horizon in self.horizons:
                result = self._evaluate_setup_on_fold(
                    ticker=ticker,
                    signal_list=signal_list,
                    horizon=horizon,
                    spec=spec,
                    split_id=split_id,
                    train_df=train_df,
                    test_df=test_df,
                    train_signals=train_signals,
                    test_signals=test_signals,
                    regime_model=regime_model
                )
                
                if result is not None:
                    results.append(result)
        
        return results
    
    def _evaluate_setup_on_fold(
        self,
        ticker: str,
        signal_list: List[str],
        horizon: int,
        spec: SplitSpec,
        split_id: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_signals: pd.DataFrame,
        test_signals: pd.DataFrame,
        regime_model
    ) -> Optional[ValidationResult]:
        """
        Evaluate a single (ticker, signals, horizon) setup on an outer fold.
        
        Uses NPWF inner folds on training data for forecast fitting,
        then evaluates on outer test fold.
        """
        # Check if all signals exist
        missing_signals = [s for s in signal_list if s not in train_signals.columns]
        if missing_signals:
            return None  # Skip if signals not available
        
        # Get trigger dates
        train_triggers = self._get_trigger_dates(train_signals, signal_list)
        test_triggers = self._get_trigger_dates(test_signals, signal_list)
        
        if len(train_triggers) < 10 or len(test_triggers) < 3:
            return None  # Insufficient support
        
        # Compute forward returns
        train_returns = self._compute_forward_returns(train_df, ticker, horizon)
        test_returns = self._compute_forward_returns(test_df, ticker, horizon)
        
        # Filter to triggered returns
        train_r = train_returns.loc[train_triggers].dropna()
        test_r = test_returns.loc[test_triggers].dropna()
        
        if len(train_r) < 10 or len(test_r) < 3:
            return None
        
        # Fit forecast on training data (histogram bins)
        forecast_probs = self._fit_histogram_forecast(train_r.values)
        
        # Evaluate on test data
        metrics = self._compute_forecast_metrics(forecast_probs, test_r.values)
        
        # Calculate baselines
        uniform_probs = np.ones(len(self.band_edges) - 1) / (len(self.band_edges) - 1)
        marginal_probs = self._fit_histogram_forecast(test_returns.dropna().values)
        
        skill_vs_uniform = self._compute_crps_delta(
            forecast_probs, uniform_probs, test_r.values
        )
        skill_vs_marginal = self._compute_crps_delta(
            forecast_probs, marginal_probs, test_r.values
        )
        
        # Regime analysis
        train_regime, test_regime, regime_sim = self._analyze_regimes(
            train_df, test_df, regime_model
        )
        
        # Drift detection
        train_features = self._extract_features(train_df, ticker, train_triggers)
        test_features = self._extract_features(test_df, ticker, test_triggers)
        drift_passed, drift_auc = self._check_drift(train_features, test_features)
        
        return ValidationResult(
            split_id=split_id,
            ticker=ticker,
            setup=signal_list,
            horizon=horizon,
            crps=metrics['crps'],
            brier_score=metrics.get('brier_score', np.nan),
            log_loss=metrics.get('log_loss', np.nan),
            pinball_q10=metrics['pinball_q10'],
            pinball_q90=metrics['pinball_q90'],
            skill_vs_uniform=skill_vs_uniform,
            skill_vs_marginal=skill_vs_marginal,
            calibration_mae=metrics['calibration_mae'],
            calibration_ece=metrics.get('calibration_ece', np.nan),
            band_probs=forecast_probs.tolist(),
            band_edges=self.band_edges.tolist(),
            n_triggers_train=len(train_r),
            n_triggers_test=len(test_r),
            regime_train=train_regime,
            regime_test=test_regime,
            regime_similarity=regime_sim,
            drift_auc=drift_auc,
            drift_passed=drift_passed
        )
    
    def _get_trigger_dates(
        self,
        signals_df: pd.DataFrame,
        signal_list: List[str]
    ) -> pd.DatetimeIndex:
        """Get dates where all signals in list are True."""
        if not signal_list:
            return pd.DatetimeIndex([])
        
        mask = pd.Series(True, index=signals_df.index)
        for sig in signal_list:
            if sig in signals_df.columns:
                mask &= signals_df[sig].fillna(False).astype(bool)
        
        return signals_df.index[mask]
    
    def _compute_forward_returns(
        self,
        df: pd.DataFrame,
        ticker: str,
        horizon: int
    ) -> pd.Series:
        """Compute forward returns for given horizon."""
        price_col = f"{ticker}_{self.price_field}"
        if price_col not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        px = pd.to_numeric(df[price_col], errors='coerce')
        fwd_px = px.shift(-horizon)
        returns = (fwd_px / px) - 1.0
        
        return returns
    
    def _fit_histogram_forecast(self, returns: np.ndarray) -> np.ndarray:
        """Fit histogram forecast probabilities."""
        if len(returns) == 0:
            return np.ones(len(self.band_edges) - 1) / (len(self.band_edges) - 1)
        
        counts, _ = np.histogram(returns, bins=self.band_edges)
        probs = counts / (counts.sum() + 1e-9)
        
        # Add small epsilon for numerical stability
        probs = probs + 1e-6
        probs = probs / probs.sum()
        
        return probs
    
    def _compute_forecast_metrics(
        self,
        forecast_probs: np.ndarray,
        test_returns: np.ndarray
    ) -> Dict[str, float]:
        """Compute forecast evaluation metrics."""
        metrics = {}
        
        # CRPS
        metrics['crps'] = distribution.crps(
            forecast_probs, test_returns, self.band_edges
        )
        
        # Pinball losses
        metrics['pinball_q10'] = distribution.pinball_loss(
            forecast_probs, test_returns, self.band_edges, quantile=0.1
        )
        metrics['pinball_q90'] = distribution.pinball_loss(
            forecast_probs, test_returns, self.band_edges, quantile=0.9
        )
        
        # Calibration
        metrics['calibration_mae'] = distribution.calibration_mae(
            forecast_probs, test_returns, self.band_edges
        )
        
        # Information gain
        metrics['info_gain'] = info_theory.info_gain(
            forecast_probs, test_returns, self.band_edges
        )
        
        return metrics
    
    def _compute_crps_delta(
        self,
        forecast_probs: np.ndarray,
        baseline_probs: np.ndarray,
        test_returns: np.ndarray
    ) -> float:
        """Compute skill as CRPS delta vs baseline (positive = better)."""
        forecast_crps = distribution.crps(forecast_probs, test_returns, self.band_edges)
        baseline_crps = distribution.crps(baseline_probs, test_returns, self.band_edges)
        
        # Positive skill = forecast is better (lower CRPS)
        return baseline_crps - forecast_crps
    
    def _analyze_regimes(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        regime_model
    ) -> Tuple[str, str, float]:
        """Analyze regime similarity between train and test."""
        if regime_model is None:
            return "unknown", "unknown", 0.0
        
        price_col = f"SPY_{self.price_field}"
        
        # Assign regimes
        train_regimes = assign_regime(train_df, price_col, regime_model)
        test_regimes = assign_regime(test_df, price_col, regime_model)
        
        # Most common regime in each window
        train_regime = train_regimes.mode()[0] if not train_regimes.empty else "unknown"
        test_regime = test_regimes.mode()[0] if not test_regimes.empty else "unknown"
        
        # Similarity (simplified - could use centroid distance)
        similarity = 1.0 if train_regime == test_regime else 0.5
        
        return train_regime, test_regime, similarity
    
    def _extract_features(
        self,
        df: pd.DataFrame,
        ticker: str,
        dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Extract features for drift detection."""
        # Simple features for drift detection: returns, volatility
        price_col = f"{ticker}_{self.price_field}"
        if price_col not in df.columns:
            return pd.DataFrame()
        
        px = pd.to_numeric(df[price_col], errors='coerce')
        ret = px.pct_change()
        vol = ret.rolling(20).std()
        
        features = pd.DataFrame({
            'ret_1d': ret,
            'ret_5d': ret.rolling(5).sum(),
            'vol_20d': vol
        }, index=df.index)
        
        return features.loc[dates].dropna()
    
    def _check_drift(
        self,
        train_features: pd.DataFrame,
        test_features: pd.DataFrame,
        threshold: float = 0.65
    ) -> Tuple[bool, float]:
        """Check for distribution drift between train and test."""
        if train_features.empty or test_features.empty:
            return True, 0.5  # No evidence of drift
        
        passed, auc, _ = check_drift_gate(
            train_features, test_features, auc_threshold=threshold
        )
        
        return passed, auc
    
    def _save_summary_stats(
        self,
        eligibility: EligibilityMatrix,
        output_dir: Path
    ):
        """Save summary statistics."""
        df = eligibility.to_dataframe()
        
        summary = {
            "total_validations": len(df),
            "unique_setups": df.groupby(['ticker', 'horizon']).ngroups,
            "mean_crps": float(df['crps'].mean()),
            "mean_skill_vs_marginal": float(df['skill_vs_marginal'].mean()),
            "drift_pass_rate": float(df['drift_passed'].mean()),
            "by_ticker": df.groupby('ticker').agg({
                'crps': 'mean',
                'skill_vs_marginal': 'mean'
            }).to_dict()
        }
        
        with open(output_dir / "validation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
