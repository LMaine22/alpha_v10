"""
Tests for objective transforms and proper scoring rules.

Verifies forecast-first objective transforms work correctly and legacy
P&L objectives can be filtered out.
"""

import pytest
import numpy as np
import sys
import importlib.util
from pathlib import Path

# Direct import to avoid circular dependencies through eval/__init__.py
_objectives_path = Path(__file__).parent.parent / "alpha_discovery" / "eval" / "objectives.py"
spec = importlib.util.spec_from_file_location("objectives", _objectives_path)
objectives_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(objectives_module)

# Import what we need
OBJECTIVE_TRANSFORMS = objectives_module.OBJECTIVE_TRANSFORMS
PROPER_SCORING_RULES = objectives_module.PROPER_SCORING_RULES
is_proper_scoring_rule = objectives_module.is_proper_scoring_rule
is_legacy_pnl_objective = objectives_module.is_legacy_pnl_objective
filter_to_proper_scoring_rules = objectives_module.filter_to_proper_scoring_rules
get_recommended_objectives = objectives_module.get_recommended_objectives
apply_objective_transforms = objectives_module.apply_objective_transforms
audit_objectives = objectives_module.audit_objectives


class TestProperScoringRules:
    """Test proper scoring rule detection and filtering."""
    
    def test_proper_scoring_rules_registered(self):
        """Verify all proper scoring rules are in OBJECTIVE_TRANSFORMS."""
        for rule in PROPER_SCORING_RULES:
            assert rule in OBJECTIVE_TRANSFORMS, (
                f"Proper scoring rule '{rule}' not in OBJECTIVE_TRANSFORMS"
            )
    
    def test_is_proper_scoring_rule_detection(self):
        """Test proper scoring rule detection."""
        # Should be proper scoring rules
        assert is_proper_scoring_rule("crps") == True
        assert is_proper_scoring_rule("brier_score") == True
        assert is_proper_scoring_rule("log_loss") == True
        assert is_proper_scoring_rule("pinball_q10") == True
        assert is_proper_scoring_rule("pinball_q90") == True
        
        # Should NOT be proper scoring rules
        assert is_proper_scoring_rule("ig_sharpe") == False
        assert is_proper_scoring_rule("min_ig") == False
        assert is_proper_scoring_rule("sharpe_ratio") == False
        assert is_proper_scoring_rule("info_gain") == False  # Info gain is not a proper scoring rule
    
    def test_is_legacy_pnl_objective_detection(self):
        """Test legacy P&L objective detection."""
        # Should be legacy P&L
        assert is_legacy_pnl_objective("ig_sharpe") == True
        assert is_legacy_pnl_objective("min_ig") == True
        assert is_legacy_pnl_objective("sharpe_ratio") == True
        assert is_legacy_pnl_objective("sortino_ratio") == True
        
        # Should NOT be legacy P&L
        assert is_legacy_pnl_objective("crps") == False
        assert is_legacy_pnl_objective("brier_score") == False
        assert is_legacy_pnl_objective("calibration_mae") == False
    
    def test_filter_to_proper_scoring_rules(self):
        """Test filtering mixed objectives to proper scoring rules only."""
        # Mixed list with proper scoring rules and legacy objectives
        mixed_objectives = [
            "crps",
            "ig_sharpe",  # Legacy - should be filtered out
            "pinball_q10",
            "min_ig",  # Legacy - should be filtered out
            "brier_score",
            "calibration_mae",  # Not proper scoring, but not legacy either
        ]
        
        filtered = filter_to_proper_scoring_rules(mixed_objectives)
        
        # Only proper scoring rules should remain
        assert "crps" in filtered
        assert "pinball_q10" in filtered
        assert "brier_score" in filtered
        
        # Legacy should be removed
        assert "ig_sharpe" not in filtered
        assert "min_ig" not in filtered
        
        # Non-proper, non-legacy should be removed too
        assert "calibration_mae" not in filtered


class TestRecommendedObjectives:
    """Test recommended objective presets."""
    
    def test_forecast_mode_only_proper_scoring(self):
        """Verify forecast mode returns only proper scoring rules."""
        objectives = get_recommended_objectives("forecast")
        
        assert len(objectives) > 0
        
        # All should be proper scoring rules
        for obj in objectives:
            assert is_proper_scoring_rule(obj), (
                f"Forecast mode returned non-proper scoring rule: {obj}"
            )
    
    def test_legacy_mode_only_pnl(self):
        """Verify legacy mode returns only P&L objectives."""
        objectives = get_recommended_objectives("legacy")
        
        assert len(objectives) > 0
        
        # All should be legacy P&L
        for obj in objectives:
            assert is_legacy_pnl_objective(obj), (
                f"Legacy mode returned non-P&L objective: {obj}"
            )
    
    def test_balanced_mode_mixed(self):
        """Verify balanced mode returns mix of objectives."""
        objectives = get_recommended_objectives("balanced")
        
        assert len(objectives) > 0
        # Just verify it returns something reasonable
        assert isinstance(objectives, list)
    
    def test_invalid_mode_raises(self):
        """Verify invalid mode raises error."""
        with pytest.raises(ValueError, match="Unknown mode"):
            get_recommended_objectives("invalid_mode")


class TestObjectiveTransforms:
    """Test objective transform application."""
    
    def test_loss_objectives_negated(self):
        """Test loss-like objectives (CRPS, Brier, etc.) are negated."""
        metrics = {
            "crps": 0.1,
            "brier_score": 0.2,
            "log_loss": 0.5,
            "pinball_q10": 0.05
        }
        
        objs, labels = apply_objective_transforms(
            metrics,
            ["crps", "brier_score", "pinball_q10"]
        )
        
        # All should be negated (lower is better → maximize negative)
        assert objs[0] == -0.1  # CRPS
        assert objs[1] == -0.2  # Brier
        assert objs[2] == -0.05  # Pinball
        
        # Labels should indicate negation
        assert labels["crps"] == "negate"
        assert labels["brier_score"] == "negate"
        assert labels["pinball_q10"] == "negate"
    
    def test_score_objectives_identity(self):
        """Test score-like objectives (info_gain, skill) are not transformed."""
        metrics = {
            "info_gain": 0.5,
            "skill_vs_marginal": 0.02
        }
        
        objs, labels = apply_objective_transforms(
            metrics,
            ["info_gain", "skill_vs_marginal"]
        )
        
        # Should be identity (higher is better → maximize as-is)
        assert objs[0] == 0.5
        assert objs[1] == 0.02
        
        # Labels should indicate identity
        assert labels["info_gain"] == "identity"
        assert labels["skill_vs_marginal"] == "identity"
    
    def test_missing_metric_raises(self):
        """Test missing metric raises KeyError (unknown objective)."""
        metrics = {"crps": 0.1}
        
        # Missing_metric is not in OBJECTIVE_TRANSFORMS, so it's an unknown objective
        with pytest.raises(KeyError, match="Unknown objective"):
            apply_objective_transforms(metrics, ["crps", "missing_metric"])
    
    def test_nan_metric_raises(self):
        """Test NaN metric raises ValueError."""
        metrics = {"crps": np.nan}
        
        with pytest.raises(ValueError, match="non-finite"):
            apply_objective_transforms(metrics, ["crps"])
    
    def test_inf_metric_raises(self):
        """Test Inf metric raises ValueError."""
        metrics = {"crps": np.inf}
        
        with pytest.raises(ValueError, match="non-finite"):
            apply_objective_transforms(metrics, ["crps"])
    
    def test_unknown_objective_raises(self):
        """Test unknown objective raises KeyError."""
        metrics = {"crps": 0.1}
        
        with pytest.raises(KeyError, match="Unknown objective"):
            apply_objective_transforms(metrics, ["unknown_objective"])


class TestLegacyFiltering:
    """Test legacy P&L objective filtering in forecast-first mode."""
    
    def test_allow_legacy_default_true(self):
        """Test legacy objectives allowed by default."""
        metrics = {"ig_sharpe": 2.0}
        
        # Should work with default allow_legacy=True
        objs, labels = apply_objective_transforms(metrics, ["ig_sharpe"])
        
        assert objs[0] == 2.0
        assert "LEGACY" in labels["ig_sharpe"]
    
    def test_allow_legacy_false_rejects(self):
        """Test legacy objectives rejected when allow_legacy=False."""
        metrics = {"ig_sharpe": 2.0}
        
        with pytest.raises(ValueError, match="Legacy P&L objective"):
            apply_objective_transforms(
                metrics,
                ["ig_sharpe"],
                allow_legacy=False
            )
    
    def test_allow_legacy_false_allows_proper_scoring(self):
        """Test proper scoring rules still allowed when allow_legacy=False."""
        metrics = {"crps": 0.1, "pinball_q10": 0.05}
        
        # Should work - these are proper scoring rules
        objs, labels = apply_objective_transforms(
            metrics,
            ["crps", "pinball_q10"],
            allow_legacy=False
        )
        
        assert len(objs) == 2
        assert objs[0] == -0.1
        assert objs[1] == -0.05


class TestAuditObjectives:
    """Test objective auditing."""
    
    def test_audit_valid_objectives(self):
        """Test audit passes for valid objectives."""
        ok, missing = audit_objectives(["crps", "pinball_q10", "info_gain"])
        
        assert ok == True
        assert len(missing) == 0
    
    def test_audit_invalid_objectives(self):
        """Test audit fails for invalid objectives."""
        ok, missing = audit_objectives(["crps", "invalid_obj1", "invalid_obj2"])
        
        assert ok == False
        assert "invalid_obj1" in missing
        assert "invalid_obj2" in missing
        assert "crps" not in missing


class TestObjectiveRegistry:
    """Test objective registry completeness."""
    
    def test_all_proper_scoring_rules_have_negate_transform(self):
        """Verify all proper scoring rules use negate transform (loss-like)."""
        for rule in PROPER_SCORING_RULES:
            transform_fn, label = OBJECTIVE_TRANSFORMS[rule]
            
            # All proper scoring rules should be negated (lower is better)
            assert label == "negate", (
                f"Proper scoring rule '{rule}' has transform '{label}', "
                f"expected 'negate'"
            )
    
    def test_coverage_objectives_have_target_transform(self):
        """Verify coverage objectives use target-based transforms."""
        coverage_objectives = ["coverage_80", "coverage_90", "coverage_95"]
        
        for obj in coverage_objectives:
            if obj in OBJECTIVE_TRANSFORMS:
                _, label = OBJECTIVE_TRANSFORMS[obj]
                assert "target" in label, (
                    f"Coverage objective '{obj}' should use target transform"
                )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
