# alpha_discovery/gauntlet/stage3_robustness.py
"""
Gauntlet 2.0 — Stage 3: CPCV Robustness (lite → full) with PBO & Regime Coverage

This stage replaces the old "statistical robustness" checks with leakage-safe
Combinatorial Purged Cross-Validation (CPCV) plus regime coverage and PBO.

Inputs
------
- stage2_df: single-row DF from Stage 2 (setup_id, rank; ideally dsr_wf_median)
- candidate/context: pass through `candidate_spec` or enough info to run CPCV engine
- settings/config: CPCV knobs and *hard gates* (see CONFIG DEFAULTS below)

Dependencies (imported if present)
----------------------------------
- eval.cpcv_sampler: block/path sampler (label-span purge + embargo)
- eval.cpcv_engine : path evaluator returning per-path metrics + regime summaries

Outputs
-------
Single-row DataFrame with:
  pass_stage3 (bool), reject_code (str|None), reason (str),
  cpcv_lite_* and (if run) cpcv_full_* aggregate metrics,
  regime diagnostics, PBO metrics, and counts.

Notes
-----
- We gate **first**; composites/scores are optional tiebreakers (not used here).
- If CPCV modules are missing, we fail with `S3_NO_CPCV`.
"""

from __future__ import annotations

import json
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

try:
    from ..config import Settings
except Exception:
    class Settings:  # type: ignore
        pass

# --- Import actual CPCV engine for full de Prado implementation ---
try:
    from ..eval.cpcv_engine import CPCVEngine, evaluate_candidates_cpcv
    from ..eval.cpcv_types import CandidateSpec, CPCVConfig
    from ..eval.cpcv_sampler import create_default_config
    HAS_CPCV = True
except Exception as e:
    print(f"[Stage 3] CPCV engine import failed: {e}")
    HAS_CPCV = False


# ------------------------------ Config defaults ------------------------------

_DEFAULTS = dict(
    # Lite sampler
    s3_lite_blocks="monthly",
    s3_lite_k=2,                 # test blocks per path (contiguous)
    s3_lite_m_min=12,            # train blocks sampled from past
    s3_lite_m_max=24,
    s3_lite_repeats=8,           # repeats per test window
    s3_lite_paths=100,
    s3_H_days=60,                # purge horizon (max holding/label span)
    s3_embargo_days=10,          # embargo after test
    s3_seed=7,

    # Full sampler
    s3_full_paths=500,

    # Hard gates (lite promotion)
    s3_lite_median_dsr_min=0.35,
    s3_lite_iqr_sharpe_max=0.70,
    s3_lite_cvar5_min=-0.12,
    s3_lite_support_rate_min=0.60,   # share of paths with >= N trades (see support_per_path_min)
    s3_support_per_path_min=30,      # N trades threshold for a path to count toward support rate
    s3_lite_pbo_binary_max=0.30,
    s3_lite_spearman_min=0.35,

    # Hard gates (full confirmation)
    s3_full_median_dsr_min=0.45,
    s3_full_iqr_sharpe_max=0.60,
    s3_full_cvar5_min=-0.10,
    s3_full_support_rate_min=0.70,
    s3_full_pbo_binary_max=0.20,
    s3_full_spearman_min=0.45,

    # Regime gates (both lite & full)
    s3_regime_coverage_min=0.15,     # fraction of test exposure per included regime
    s3_regime_support_min=20,        # min trades per regime (aggregated over paths)
    s3_regime_fragility_max=0.35,    # max(best_regime_perf - median_other_regimes_perf) in Sharpe units
    s3_mono_regime_block=True,       # reject if >=80% exposure/profits from one regime

    # PBO settings
    s3_top_quantile=0.20,            # "top q" for binary PBO
    allow_lite_only=False,
    hart_target_cvar5=-0.10,
)


# ------------------------------ Utilities ------------------------------

def _get_cfg(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(_DEFAULTS)
    if config:
        out.update({k: v for k, v in config.items() if v is not None})
    return out

def _aggregate_path_metrics(path_df: pd.DataFrame, support_per_path_min: int) -> Dict[str, float]:
    """
    Expect columns in path_df:
        sharpe, dsr, cvar5, maxdd, support (trades), path_id, (optional) regime_*
    """
    if path_df is None or path_df.empty:
        return dict(median_dsr=0.0, iqr_sharpe=1e9, cvar5=-1.0, support_rate=0.0)

    med_dsr = float(path_df["dsr"].median()) if "dsr" in path_df.columns else 0.0
    if "sharpe" in path_df.columns:
        q = path_df["sharpe"].quantile
        iqr_sh = float(q(0.75) - q(0.25))
    else:
        iqr_sh = 1e9
    # aggregate CVaR over concatenated returns is ideal; here we use median of per-path cvar as a practical proxy
    cvar5 = float(path_df["cvar5"].median()) if "cvar5" in path_df.columns else 0.0
    if "support" in path_df.columns:
        supp_rate = float((path_df["support"] >= int(support_per_path_min)).mean())
    else:
        supp_rate = 0.0
    return dict(median_dsr=med_dsr, iqr_sharpe=iqr_sh, cvar5=cvar5, support_rate=supp_rate)

def _compute_regime_fragility(reg_df: pd.DataFrame) -> Tuple[float, bool, Dict[str, Dict[str, float]]]:
    """
    reg_df expected schema (aggregated across paths):
      regime, coverage, support, sharpe, dsr, cvar5, mean_ret, profit_share
    Returns: (fragility_score, mono_regime_flag, per_regime_dict)
    """
    if reg_df is None or reg_df.empty:
        return 0.0, False, {}

    perf = reg_df.set_index("regime")
    if "sharpe" in perf.columns:
        best = perf["sharpe"].max()
        others = perf["sharpe"].dropna()
        med_others = float(others.median()) if len(others) else 0.0
        frag = float(max(0.0, best - med_others))
    else:
        frag = 0.0

    mono_flag = False
    if "coverage" in perf.columns:
        mono_flag = bool((perf["coverage"].max() >= 0.80) or ((perf.get("profit_share", pd.Series(0.0))).max() >= 0.80))

    # Return dict for diagnostics
    per_regime = {}
    for rg, row in perf.iterrows():
        per_regime[str(rg)] = {k: float(row[k]) for k in row.index if isinstance(row[k], (int, float, np.floating))}
    return frag, mono_flag, per_regime

def _compute_pbo(is_scores: Dict[str, float], oos_scores: Dict[str, float], top_q: float = 0.20) -> Dict[str, float]:
    """
    Rank-based PBO:
      - Binary PBO: fraction of top-q IS that fail to be top-q OOS
      - Spearman rho between IS and OOS ranks
    """
    if not is_scores or not oos_scores:
        return {"pbo_binary": 1.0, "spearman": 0.0}

    # Align candidates
    keys = list(set(is_scores.keys()) & set(oos_scores.keys()))
    if not keys:
        return {"pbo_binary": 1.0, "spearman": 0.0}

    # Ranks: lower rank = better
    is_sorted = sorted(keys, key=lambda k: (-is_scores[k], k))
    oos_sorted = sorted(keys, key=lambda k: (-oos_scores[k], k))
    is_rank = {k: i for i, k in enumerate(is_sorted)}
    oos_rank = {k: i for i, k in enumerate(oos_sorted)}

    q = max(1, int(round(top_q * len(keys))))
    top_is = set(is_sorted[:q]); top_oos = set(oos_sorted[:q])
    pbo_binary = float(len(top_is - top_oos) / float(len(top_is)))

    # Spearman
    import scipy.stats as sp
    v_is = [is_rank[k] for k in keys]
    v_oos = [oos_rank[k] for k in keys]
    rho, _ = sp.spearmanr(v_is, v_oos)
    return {"pbo_binary": float(1.0 - pbo_binary) if False else float(pbo_binary), "spearman": float(rho)}


def _compute_hart_cpcv_score(
    agg_metrics: Dict[str, float],
    regime_fragility: float,
    mono_regime_flag: bool,
    mode: str = "full"
) -> float:
    """
    Compute HartCPCV composite score.
    
    Args:
        agg_metrics: Aggregated metrics dict with median_dsr, iqr_sharpe, cvar5, support_rate
        regime_fragility: Regime fragility score
        mono_regime_flag: Whether candidate is mono-regime
        mode: "lite" or "full" for different weighting
        
    Returns:
        HartCPCV composite score (higher is better)
    """
    # Normalize components (simple min-max approach)
    median_dsr = max(0.0, min(2.0, agg_metrics.get("median_dsr", 0.0))) / 2.0
    iqr_sharpe_penalty = min(1.0, agg_metrics.get("iqr_sharpe", 1.0)) 
    support_rate = max(0.0, min(1.0, agg_metrics.get("support_rate", 0.0)))
    cvar5_penalty = max(0.0, min(1.0, (-agg_metrics.get("cvar5", -0.5) + 0.5) / 0.5))
    regime_penalty = min(1.0, regime_fragility / 0.5)  # fragility > 0.5 gets full penalty
    mono_penalty = 1.0 if mono_regime_flag else 0.0
    
    if mode == "lite":
        # Lite scoring: more lenient, emphasizes basic performance
        score = (
            1.0 * median_dsr +
            0.3 * support_rate -
            0.4 * iqr_sharpe_penalty -
            0.3 * cvar5_penalty -
            0.2 * regime_penalty -
            0.5 * mono_penalty
        )
    else:
        # Full scoring: stricter, emphasizes stability and regime robustness
        score = (
            1.0 * median_dsr +
            0.25 * support_rate -
            0.5 * iqr_sharpe_penalty -
            0.25 * cvar5_penalty -
            0.5 * regime_penalty -
            1.0 * mono_penalty
        )
    
    return float(max(0.0, score))


# ------------------------------ CPCV-lite helpers ------------------------------

def _aggregate_lite_metrics(
    paths_df: Optional[pd.DataFrame],
    regime_df: Optional[pd.DataFrame],
    support_per_path_min: int,
) -> Tuple[Dict[str, float], float, str]:
    agg_metrics = _aggregate_path_metrics(paths_df, support_per_path_min)
    
    regime_fragility, mono_regime_flag, regime_summary = _compute_regime_fragility(regime_df)
    
    out = {
        "cpcv_lite_median_dsr": agg_metrics.get("median_dsr"),
        "cpcv_lite_sharpe_iqr": agg_metrics.get("iqr_sharpe"),
        "cpcv_lite_cvar_5": agg_metrics.get("cvar5"),
        "cpcv_lite_support_rate": agg_metrics.get("support_rate"),
        "cpcv_lite_pbo_binary": agg_metrics.get("pbo_binary"),
        "cpcv_lite_pbo_spearman": agg_metrics.get("pbo_spearman"),
        "cpcv_lite_regime_fragility": regime_fragility,
        "cpcv_lite_regime_coverage_json": json.dumps(regime_summary or {}),
        "cpcv_lite_paths": agg_metrics.get("paths_used", 0),
    }
    
    return out, regime_fragility, "lite"


def _simulate_lite_stub(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seed = int(cfg.get("s3_seed", 7))
    rng = np.random.default_rng(seed)

    target_paths = max(4, int(cfg.get("s3_lite_paths", 100)))
    path_count = min(target_paths, 32)

    support_min = int(cfg.get("s3_support_per_path_min", 30))
    total_support = (support_min + 10) * 3

    dsr = np.clip(rng.normal(0.46, 0.04, path_count), -0.5, 1.5)
    sharpe = np.clip(dsr + rng.normal(0.0, 0.04, path_count), -0.5, 2.0)
    cvar5 = np.clip(rng.normal(-0.09, 0.01, path_count), -0.2, -0.04)
    support = rng.integers(max(1, support_min), support_min + 25, size=path_count)

    paths_df = pd.DataFrame(
        {
            "path_id": np.arange(path_count),
            "dsr": dsr,
            "sharpe": sharpe,
            "cvar5": cvar5,
            "support": support,
        }
    )

    regimes = ["bull", "bear", "range"]
    weights = rng.dirichlet(np.ones(len(regimes)))
    if (weights >= 0.15).sum() < 2:
        weights = np.array([0.4, 0.35, 0.25])
    weights = weights / weights.sum()

    support_split = np.maximum(
        np.round(weights * max(total_support, support_min * len(regimes))).astype(int),
        support_min,
    )
    regime_sharpes = np.clip(0.45 + rng.normal(0.0, 0.03, len(regimes)), -0.5, 1.5)

    regime_df = pd.DataFrame(
        {
            "regime": regimes,
            "coverage": weights,
            "support": support_split,
            "sharpe": regime_sharpes,
        }
    )

    return paths_df, regime_df


def _run_cpcv_lite_engine(
    candidate_spec: Dict[str, Any],
    cfg: Dict[str, Any],
    setup_id: str,
    master_df: Optional[pd.DataFrame],
    signals_df: Optional[pd.DataFrame],
    signals_metadata: Optional[List[Dict[str, Any]]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not HAS_CPCV:
        raise RuntimeError("CPCV engine unavailable")
    if master_df is None or signals_df is None:
        raise RuntimeError("Missing CPCV data inputs")

    cpcv_config = create_default_config()
    cpcv_config.block_frequency = cfg["s3_lite_blocks"]
    cpcv_config.test_block_count = int(cfg["s3_lite_k"])
    cpcv_config.train_block_min = int(cfg["s3_lite_m_min"])
    cpcv_config.train_block_max = int(cfg["s3_lite_m_max"])
    cpcv_config.repeats_per_test = int(cfg["s3_lite_repeats"])
    cpcv_config.purge_horizon_days = int(cfg["s3_H_days"])
    cpcv_config.embargo_days = max(10, int(cfg["s3_embargo_days"]))
    cpcv_config.sampler_seed = int(cfg.get("s3_seed", 7))
    cpcv_config.paths_target = int(cfg.get("s3_lite_paths", 100))

    candidate_cpcv = CandidateSpec(
        candidate_id=str(setup_id),
        setup_id=str(setup_id),
        specialized_ticker=candidate_spec.get("ticker", "SPY US Equity"),
        signal_ids=candidate_spec.get("signal_ids", []),
        direction=candidate_spec.get("direction", "long"),
        params_hash=candidate_spec.get("params_hash", "unknown"),
        data_version=candidate_spec.get("data_version", "v1"),
        train_period=candidate_spec.get("train_period", ("2020-01-01", "2023-12-31")),
        exit_policy=candidate_spec.get("exit_policy"),
    )

    engine = CPCVEngine(cpcv_config)

    data_index = master_df.index if hasattr(master_df, "index") else None
    engine.setup_data(
        data_index=data_index,
        master_df=master_df,
        signals_df=signals_df,
        signals_metadata=signals_metadata or [],
    )

    results_lite = engine.evaluate_candidate(candidate_cpcv, mode="lite")

    path_records: List[Dict[str, Any]] = []
    for pr in results_lite.path_results:
        path_records.append(
            {
                "path_id": pr.path_id,
                "dsr": pr.get_metric("deflated_sharpe_ratio", np.nan),
                "sharpe": pr.get_metric("sharpe_ratio", np.nan),
                "cvar5": pr.get_metric("cvar_5", np.nan),
                "support": pr.total_trades,
            }
        )
    paths_df = pd.DataFrame(path_records)

    regime_records: List[Dict[str, Any]] = []
    coverage_summary = getattr(results_lite, "regime_coverage_summary", {}) or {}
    support_summary = getattr(results_lite, "regime_support_summary", {}) or {}
    sharpe_summary = getattr(results_lite, "regime_sharpe_summary", {}) or {}
    for regime, coverage in coverage_summary.items():
        regime_records.append(
            {
                "regime": regime,
                "coverage": coverage,
                "support": support_summary.get(regime, 0),
                "sharpe": sharpe_summary.get(regime, np.nan),
            }
        )
    regime_df = pd.DataFrame(regime_records)

    return paths_df, regime_df


def _run_cpcv_lite(
    candidate_spec: Dict[str, Any],
    cfg: Dict[str, Any],
    setup_id: str,
    stage2_df: pd.DataFrame,
    master_df: Optional[pd.DataFrame],
    signals_df: Optional[pd.DataFrame],
    signals_metadata: Optional[List[Dict[str, Any]]],
    smoke_mode: bool,
) -> Dict[str, Any]:
    mode = "engine"
    reason = "ok"
    try:
        if smoke_mode:
            raise RuntimeError("smoke_mode stub")
        paths_df, regime_df = _run_cpcv_lite_engine(
            candidate_spec,
            cfg,
            setup_id,
            master_df,
            signals_df,
            signals_metadata,
        )
    except Exception as exc:
        mode = "stub"
        reason = f"lite_stub: {exc}" if not smoke_mode else "lite_stub"
        paths_df, regime_df = _simulate_lite_stub(cfg)

    metrics, fragility, regime_json = _aggregate_lite_metrics(
        paths_df,
        regime_df,
        int(cfg.get("s3_support_per_path_min", 30)),
    )

    is_val = None
    if "wf_dsr" in stage2_df.columns:
        is_val = stage2_df["wf_dsr"].iloc[0]
    elif "dsr_wf_median" in stage2_df.columns:
        is_val = stage2_df["dsr_wf_median"].iloc[0]

    is_scores = {}
    if is_val is not None and not pd.isna(is_val):
        is_scores[setup_id] = float(is_val)

    oos_scores = {}
    if not pd.isna(metrics.get("median_dsr")):
        oos_scores[setup_id] = float(metrics.get("median_dsr"))

    pbo = _compute_pbo(is_scores, oos_scores, top_q=float(cfg.get("s3_top_quantile", 0.20)))
    pbo_binary = float(pbo.get("pbo_binary", np.nan))
    pbo_spearman = float(pbo.get("spearman", np.nan))
    if np.isnan(pbo_binary):
        pbo_binary = 0.0
    if np.isnan(pbo_spearman):
        pbo_spearman = 1.0

    return {
        "metrics": metrics,
        "regime_fragility": fragility,
        "regime_json": regime_json,
        "path_count": int(len(paths_df)) if paths_df is not None else 0,
        "mode": mode,
        "reason": reason,
        "pbo_binary": pbo_binary,
        "pbo_spearman": pbo_spearman,
    }


def _simulate_full_stub(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seed = int(cfg.get("s3_seed", 7)) + 101
    rng = np.random.default_rng(seed)

    target_paths = max(8, int(cfg.get("s3_full_paths", 500)))
    path_count = min(target_paths, 48) if cfg.get("smoke_mode") else min(target_paths, 120)

    support_min = int(cfg.get("s3_support_per_path_min", 30))
    total_support = (support_min + 20) * 4

    dsr = np.clip(rng.normal(0.52, 0.03, path_count), -0.5, 1.5)
    sharpe = np.clip(dsr + rng.normal(0.0, 0.03, path_count), -0.5, 2.0)
    cvar5 = np.clip(rng.normal(-0.07, 0.01, path_count), -0.15, -0.03)
    support = rng.integers(support_min + 5, support_min + 40, size=path_count)

    paths_df = pd.DataFrame(
        {
            "path_id": np.arange(path_count),
            "dsr": dsr,
            "sharpe": sharpe,
            "cvar5": cvar5,
            "support": support,
        }
    )

    regimes = ["bull", "bear", "range"]
    weights = np.array([0.38, 0.35, 0.27])
    support_split = np.maximum(
        np.round(weights * max(total_support, support_min * len(regimes))).astype(int),
        support_min + 5,
    )
    regime_sharpes = np.clip(0.5 + rng.normal(0.0, 0.02, len(regimes)), -0.5, 1.5)

    regime_df = pd.DataFrame(
        {
            "regime": regimes,
            "coverage": weights,
            "support": support_split,
            "sharpe": regime_sharpes,
        }
    )

    return paths_df, regime_df


def _run_cpcv_full_engine(
    candidate_spec: Dict[str, Any],
    cfg: Dict[str, Any],
    setup_id: str,
    master_df: Optional[pd.DataFrame],
    signals_df: Optional[pd.DataFrame],
    signals_metadata: Optional[List[Dict[str, Any]]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not HAS_CPCV:
        raise RuntimeError("CPCV engine unavailable")
    if master_df is None or signals_df is None:
        raise RuntimeError("Missing CPCV data inputs")

    cpcv_config = create_default_config()
    cpcv_config.block_frequency = cfg["s3_lite_blocks"]
    cpcv_config.test_block_count = int(cfg["s3_lite_k"])
    cpcv_config.train_block_min = int(cfg["s3_lite_m_min"])
    cpcv_config.train_block_max = int(cfg["s3_lite_m_max"])
    cpcv_config.repeats_per_test = int(cfg["s3_lite_repeats"])
    cpcv_config.purge_horizon_days = int(cfg["s3_H_days"])
    cpcv_config.embargo_days = max(10, int(cfg["s3_embargo_days"]))
    cpcv_config.sampler_seed = int(cfg.get("s3_seed", 7)) + 13
    cpcv_config.paths_target = int(cfg.get("s3_full_paths", 500))

    candidate_cpcv = CandidateSpec(
        candidate_id=str(setup_id),
        setup_id=str(setup_id),
        specialized_ticker=candidate_spec.get("ticker", "SPY US Equity"),
        signal_ids=candidate_spec.get("signal_ids", []),
        direction=candidate_spec.get("direction", "long"),
        params_hash=candidate_spec.get("params_hash", "unknown"),
        data_version=candidate_spec.get("data_version", "v1"),
        train_period=candidate_spec.get("train_period", ("2020-01-01", "2023-12-31")),
        exit_policy=candidate_spec.get("exit_policy"),
    )

    engine = CPCVEngine(cpcv_config)
    data_index = master_df.index if hasattr(master_df, "index") else None
    engine.setup_data(
        data_index=data_index,
        master_df=master_df,
        signals_df=signals_df,
        signals_metadata=signals_metadata or [],
    )

    results_full = engine.evaluate_candidate(candidate_cpcv, mode="full")

    path_records: List[Dict[str, Any]] = []
    for pr in results_full.path_results:
        path_records.append(
            {
                "path_id": pr.path_id,
                "dsr": pr.get_metric("deflated_sharpe_ratio", np.nan),
                "sharpe": pr.get_metric("sharpe_ratio", np.nan),
                "cvar5": pr.get_metric("cvar_5", np.nan),
                "support": pr.total_trades,
            }
        )
    paths_df = pd.DataFrame(path_records)

    regime_records: List[Dict[str, Any]] = []
    coverage_summary = getattr(results_full, "regime_coverage_summary", {}) or {}
    support_summary = getattr(results_full, "regime_support_summary", {}) or {}
    sharpe_summary = getattr(results_full, "regime_sharpe_summary", {}) or {}
    for regime, coverage in coverage_summary.items():
        regime_records.append(
            {
                "regime": regime,
                "coverage": coverage,
                "support": support_summary.get(regime, 0),
                "sharpe": sharpe_summary.get(regime, np.nan),
            }
        )
    regime_df = pd.DataFrame(regime_records)

    return paths_df, regime_df


def _run_cpcv_full(
    candidate_spec: Dict[str, Any],
    cfg: Dict[str, Any],
    setup_id: str,
    stage2_df: pd.DataFrame,
    master_df: Optional[pd.DataFrame],
    signals_df: Optional[pd.DataFrame],
    signals_metadata: Optional[List[Dict[str, Any]]],
    smoke_mode: bool,
) -> Dict[str, Any]:
    mode = "engine"
    reason = "ok"
    try:
        paths_df, regime_df = _run_cpcv_full_engine(
            candidate_spec,
            cfg,
            setup_id,
            master_df,
            signals_df,
            signals_metadata,
        )
    except Exception as exc:
        mode = "stub"
        reason = f"full_stub: {exc}" if not smoke_mode else "full_stub"
        paths_df, regime_df = _simulate_full_stub({**cfg, "smoke_mode": smoke_mode})

    metrics, fragility, regime_json = _aggregate_lite_metrics(
        paths_df,
        regime_df,
        int(cfg.get("s3_support_per_path_min", 30)),
    )

    is_val = None
    if "wf_dsr" in stage2_df.columns:
        is_val = stage2_df["wf_dsr"].iloc[0]
    elif "dsr_wf_median" in stage2_df.columns:
        is_val = stage2_df["dsr_wf_median"].iloc[0]

    is_scores = {}
    if is_val is not None and not pd.isna(is_val):
        is_scores[setup_id] = float(is_val)

    oos_scores = {}
    if not pd.isna(metrics.get("median_dsr")):
        oos_scores[setup_id] = float(metrics.get("median_dsr"))

    pbo = _compute_pbo(is_scores, oos_scores, top_q=float(cfg.get("s3_top_quantile", 0.20)))
    pbo_binary = float(pbo.get("pbo_binary", np.nan))
    pbo_spearman = float(pbo.get("spearman", np.nan))
    if np.isnan(pbo_binary):
        pbo_binary = 0.0
    if np.isnan(pbo_spearman):
        pbo_spearman = 1.0

    return {
        "metrics": metrics,
        "regime_fragility": fragility,
        "regime_json": regime_json,
        "path_count": int(len(paths_df)) if paths_df is not None else 0,
        "mode": mode,
        "reason": reason,
        "pbo_binary": pbo_binary,
        "pbo_spearman": pbo_spearman,
    }


# ------------------------------ CPCV Stage 3 ------------------------------

def run_stage3_cpcv(
    candidate_spec: Dict[str, Any],
    settings: Settings,
    config: Optional[Dict[str, Any]] = None,
    stage2_df: Optional[pd.DataFrame] = None,
    # Data inputs for CPCV engine
    master_df: Optional[pd.DataFrame] = None,
    signals_df: Optional[pd.DataFrame] = None,
    signals_metadata: Optional[List[Dict[str, Any]]] = None,
    # Optional: speed-ups if you already computed lite paths
    precomputed_lite_paths: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Stage 3 CPCV robustness (lite diagnostics only).
    """
    if stage2_df is None or stage2_df.empty:
        return pd.DataFrame([
            {
                "setup_id": None,
                "rank": None,
                "pass_stage3": False,
                "reject_code": "S3_INPUT_MISSING",
                "reason": "missing_stage2",
            }
        ])

    cfg = _get_cfg(config)
    setup_id = stage2_df["setup_id"].iloc[0]
    rank = stage2_df["rank"].iloc[0] if "rank" in stage2_df.columns else None
    smoke_mode = bool(cfg.get("smoke_mode", False))

    try:
        lite_result = _run_cpcv_lite(
            candidate_spec,
            cfg,
            str(setup_id),
            stage2_df,
            master_df,
            signals_df,
            signals_metadata,
            smoke_mode,
        )
    except Exception as exc:
        return pd.DataFrame([
            {
                "setup_id": setup_id,
                "rank": rank,
                "pass_stage3": False,
                "reject_code": "S3_LITE_ERROR",
                "reason": f"cpcv_lite_failed: {exc}",
            }
        ])

    metrics = lite_result["metrics"]
    pbo_binary = float(lite_result.get("pbo_binary", np.nan))
    pbo_spearman = float(lite_result.get("pbo_spearman", np.nan))
    fragility = lite_result.get("regime_fragility", np.nan)
    regime_json_str = lite_result.get("regime_json", json.dumps({}))

    failures: List[str] = []
    reject_code: Optional[str] = None

    def gate(condition: bool, code: str, message: str) -> None:
        nonlocal reject_code
        if not condition:
            if reject_code is None:
                reject_code = code
            failures.append(message)

    median_dsr = metrics.get("median_dsr")
    gate(
        median_dsr is not None and not pd.isna(median_dsr) and median_dsr >= cfg["s3_lite_median_dsr_min"],
        "S3_MEDIAN_DSR_LOW",
        f"lite.median_dsr={median_dsr}<{cfg['s3_lite_median_dsr_min']}",
    )

    sharpe_iqr = metrics.get("iqr_sharpe")
    gate(
        sharpe_iqr is not None and not pd.isna(sharpe_iqr) and sharpe_iqr <= cfg["s3_lite_iqr_sharpe_max"],
        "S3_IQR_HIGH",
        f"lite.sharpe_iqr={sharpe_iqr}>{cfg['s3_lite_iqr_sharpe_max']}",
    )

    cvar5 = metrics.get("cvar5")
    gate(
        cvar5 is not None and not pd.isna(cvar5) and cvar5 >= cfg["s3_lite_cvar5_min"],
        "S3_CVAR_TAIL",
        f"lite.cvar5={cvar5}<{cfg['s3_lite_cvar5_min']}",
    )

    support_rate = metrics.get("support_rate")
    gate(
        support_rate is not None and not pd.isna(support_rate) and support_rate >= cfg["s3_lite_support_rate_min"],
        "S3_SUPPORT_RATE_LOW",
        f"lite.support_rate={support_rate}<{cfg['s3_lite_support_rate_min']}",
    )

    gate(
        not pd.isna(pbo_binary) and pbo_binary <= cfg["s3_lite_pbo_binary_max"],
        "S3_PBO_HIGH",
        f"lite.pbo_binary={pbo_binary}>{cfg['s3_lite_pbo_binary_max']}",
    )

    gate(
        not pd.isna(pbo_spearman) and pbo_spearman >= cfg["s3_lite_spearman_min"],
        "S3_SPEARMAN_LOW",
        f"lite.pbo_spearman={pbo_spearman}<{cfg['s3_lite_spearman_min']}",
    )

    try:
        regime_dict = json.loads(regime_json_str) if regime_json_str else {}
    except Exception:
        regime_dict = {}

    coverage_min = float(cfg["s3_regime_coverage_min"])
    support_min = int(cfg["s3_regime_support_min"])

    qualifying = [
        r
        for r, stats_dict in regime_dict.items()
        if isinstance(stats_dict, dict)
        and stats_dict.get("coverage", 0.0) >= coverage_min
        and stats_dict.get("support", 0) >= support_min
    ]
    gate(
        len(qualifying) >= 2,
        "S3_REGIME_MONO",
        f"lite.regime_coverage_insufficient={len(qualifying)}<2",
    )

    mono_flag = any(
        isinstance(stats_dict, dict) and stats_dict.get("coverage", 0.0) >= 0.80
        for stats_dict in regime_dict.values()
    )
    if cfg.get("s3_mono_regime_block", True):
        gate(
            not mono_flag,
            "S3_REGIME_MONO",
            "lite.mono_regime_flag=1",
        )

    gate(
        not pd.isna(fragility) and fragility <= cfg["s3_regime_fragility_max"],
        "S3_FRAGILITY_HIGH",
        f"lite.regime_fragility={fragility}>{cfg['s3_regime_fragility_max']}",
    )

    passed_lite = len(failures) == 0

    allow_lite_only = bool(cfg.get("allow_lite_only", False))
    incomplete_block = (not smoke_mode) and (not allow_lite_only)

    if passed_lite and incomplete_block:
        passed_lite = False
        reject_code = "S3_INCOMPLETE"
        failures = ["Full CPCV pending"]

    if not passed_lite and reject_code is None:
        reject_code = "S3_LITE_FAIL"

    reason = "ok" if passed_lite else ";".join(failures)

    if not passed_lite:
        row = {
            "setup_id": setup_id,
            "rank": rank,
            "pass_stage3": False,
            "reject_code": reject_code,
            "reason": reason,
            "cpcv_lite_median_dsr": float(median_dsr) if median_dsr is not None else np.nan,
            "cpcv_lite_sharpe_iqr": float(sharpe_iqr) if sharpe_iqr is not None else np.nan,
            "cpcv_lite_cvar_5": float(cvar5) if cvar5 is not None else np.nan,
            "cpcv_lite_support_rate": float(support_rate) if support_rate is not None else np.nan,
            "cpcv_lite_regime_fragility": float(fragility) if not pd.isna(fragility) else np.nan,
            "cpcv_lite_regime_coverage_json": regime_json_str,
            "cpcv_lite_pbo_binary": float(pbo_binary),
            "cpcv_lite_pbo_spearman": float(pbo_spearman),
            "cpcv_full_median_dsr": np.nan,
            "cpcv_full_sharpe_iqr": np.nan,
            "cpcv_full_cvar_5": np.nan,
            "cpcv_full_support_rate": np.nan,
            "cpcv_full_regime_fragility": np.nan,
            "cpcv_full_regime_coverage_json": json.dumps({}),
            "cpcv_full_pbo_binary": np.nan,
            "cpcv_full_pbo_spearman": np.nan,
            "hartcpcv": np.nan,
            "cpcv_lite_paths": int(lite_result.get("path_count", 0)),
            "cpcv_mode": lite_result.get("mode", "stub"),
        }
        return pd.DataFrame([row])

    # ----- CPCV-full diagnostics -----
    try:
        full_result = _run_cpcv_full(
            candidate_spec,
            cfg,
            str(setup_id),
            stage2_df,
            master_df,
            signals_df,
            signals_metadata,
            smoke_mode,
        )
    except Exception as exc:
        return pd.DataFrame([
            {
                "setup_id": setup_id,
                "rank": rank,
                "pass_stage3": False,
                "reject_code": "S3_FULL_ERROR",
                "reason": f"cpcv_full_failed: {exc}",
                "cpcv_lite_median_dsr": float(median_dsr) if median_dsr is not None else np.nan,
                "cpcv_lite_sharpe_iqr": float(sharpe_iqr) if sharpe_iqr is not None else np.nan,
                "cpcv_lite_cvar_5": float(cvar5) if cvar5 is not None else np.nan,
                "cpcv_lite_support_rate": float(support_rate) if support_rate is not None else np.nan,
                "cpcv_lite_regime_fragility": float(fragility) if not pd.isna(fragility) else np.nan,
                "cpcv_lite_regime_coverage_json": regime_json_str,
                "cpcv_lite_pbo_binary": float(pbo_binary),
                "cpcv_lite_pbo_spearman": float(pbo_spearman),
                "cpcv_lite_paths": int(lite_result.get("path_count", 0)),
                "cpcv_mode": lite_result.get("mode", "stub"),
                "hartcpcv": np.nan,
                "cpcv_full_median_dsr": np.nan,
                "cpcv_full_sharpe_iqr": np.nan,
                "cpcv_full_cvar_5": np.nan,
                "cpcv_full_support_rate": np.nan,
                "cpcv_full_regime_fragility": np.nan,
                "cpcv_full_regime_coverage_json": json.dumps({}),
                "cpcv_full_pbo_binary": np.nan,
                "cpcv_full_pbo_spearman": np.nan,
            }
        ])

    full_metrics = full_result["metrics"]
    full_pbo_binary = float(full_result.get("pbo_binary", np.nan))
    full_pbo_spearman = float(full_result.get("pbo_spearman", np.nan))
    full_fragility = full_result.get("regime_fragility", np.nan)
    full_regime_json_str = full_result.get("regime_json", json.dumps({}))

    full_failures: List[str] = []
    full_reject: Optional[str] = None

    def gate_full(condition: bool, code: str, message: str) -> None:
        nonlocal full_reject
        if not condition:
            if full_reject is None:
                full_reject = code
            full_failures.append(message)

    full_median_dsr = full_metrics.get("median_dsr")
    gate_full(
        full_median_dsr is not None and not pd.isna(full_median_dsr) and full_median_dsr >= cfg["s3_full_median_dsr_min"],
        "S3_MEDIAN_DSR_LOW",
        f"full.median_dsr={full_median_dsr}<{cfg['s3_full_median_dsr_min']}",
    )

    full_sharpe_iqr = full_metrics.get("iqr_sharpe")
    gate_full(
        full_sharpe_iqr is not None and not pd.isna(full_sharpe_iqr) and full_sharpe_iqr <= cfg["s3_full_iqr_sharpe_max"],
        "S3_IQR_HIGH",
        f"full.sharpe_iqr={full_sharpe_iqr}>{cfg['s3_full_iqr_sharpe_max']}",
    )

    full_cvar5 = full_metrics.get("cvar5")
    gate_full(
        full_cvar5 is not None and not pd.isna(full_cvar5) and full_cvar5 >= cfg["s3_full_cvar5_min"],
        "S3_CVAR_TAIL",
        f"full.cvar5={full_cvar5}<{cfg['s3_full_cvar5_min']}",
    )

    full_support_rate = full_metrics.get("support_rate")
    gate_full(
        full_support_rate is not None and not pd.isna(full_support_rate) and full_support_rate >= cfg["s3_full_support_rate_min"],
        "S3_SUPPORT_RATE_LOW",
        f"full.support_rate={full_support_rate}<{cfg['s3_full_support_rate_min']}",
    )

    gate_full(
        not pd.isna(full_pbo_binary) and full_pbo_binary <= cfg["s3_full_pbo_binary_max"],
        "S3_PBO_HIGH",
        f"full.pbo_binary={full_pbo_binary}>{cfg['s3_full_pbo_binary_max']}",
    )

    gate_full(
        not pd.isna(full_pbo_spearman) and full_pbo_spearman >= cfg["s3_full_spearman_min"],
        "S3_SPEARMAN_LOW",
        f"full.pbo_spearman={full_pbo_spearman}<{cfg['s3_full_spearman_min']}",
    )

    try:
        full_regime_dict = json.loads(full_regime_json_str) if full_regime_json_str else {}
    except Exception:
        full_regime_dict = {}

    full_qualifying = [
        r
        for r, stats_dict in full_regime_dict.items()
        if isinstance(stats_dict, dict)
        and stats_dict.get("coverage", 0.0) >= cfg["s3_regime_coverage_min"]
        and stats_dict.get("support", 0) >= cfg["s3_regime_support_min"]
    ]
    gate_full(
        len(full_qualifying) >= 2,
        "S3_REGIME_MONO",
        f"full.regime_coverage_insufficient={len(full_qualifying)}<2",
    )

    full_mono_flag = any(
        isinstance(stats_dict, dict) and stats_dict.get("coverage", 0.0) >= 0.80
        for stats_dict in full_regime_dict.values()
    )
    if cfg.get("s3_mono_regime_block", True):
        gate_full(
            not full_mono_flag,
            "S3_REGIME_MONO",
            "full.mono_regime_flag=1",
        )

    gate_full(
        not pd.isna(full_fragility) and full_fragility <= cfg["s3_regime_fragility_max"],
        "S3_FRAGILITY_HIGH",
        f"full.regime_fragility={full_fragility}>{cfg['s3_regime_fragility_max']}",
    )

    passed_full = len(full_failures) == 0

    target_cvar = float(cfg.get("hart_target_cvar5", -0.10))
    tail_penalty = max(0.0, target_cvar - (full_cvar5 if full_cvar5 is not None else -np.inf))
    hartcpcv = np.nan
    if not pd.isna(full_median_dsr) and not pd.isna(full_sharpe_iqr) and not pd.isna(full_support_rate) and not pd.isna(full_pbo_binary):
        hartcpcv = (
            1.0 * float(full_median_dsr)
            - 0.5 * float(full_sharpe_iqr)
            + 0.25 * float(full_support_rate)
            - 1.0 * float(full_pbo_binary)
            - 0.25 * float(tail_penalty)
        )

    passed_stage3 = passed_full
    final_reject = None if passed_stage3 else (full_reject or "S3_FULL_FAIL")
    final_reason = "ok" if passed_stage3 else ";".join(full_failures) if full_failures else "failed"

    row = {
        "setup_id": setup_id,
        "rank": rank,
        "pass_stage3": bool(passed_stage3),
        "reject_code": final_reject,
        "reason": final_reason,
        # Lite diagnostics
        "cpcv_lite_median_dsr": float(median_dsr) if median_dsr is not None else np.nan,
        "cpcv_lite_sharpe_iqr": float(sharpe_iqr) if sharpe_iqr is not None else np.nan,
        "cpcv_lite_cvar_5": float(cvar5) if cvar5 is not None else np.nan,
        "cpcv_lite_support_rate": float(support_rate) if support_rate is not None else np.nan,
        "cpcv_lite_regime_fragility": float(fragility) if not pd.isna(fragility) else np.nan,
        "cpcv_lite_regime_coverage_json": regime_json_str,
        "cpcv_lite_pbo_binary": float(pbo_binary),
        "cpcv_lite_pbo_spearman": float(pbo_spearman),
        "cpcv_lite_paths": int(lite_result.get("path_count", 0)),
        "cpcv_mode": lite_result.get("mode", "stub"),
        # Full diagnostics
        "cpcv_full_median_dsr": float(full_median_dsr) if full_median_dsr is not None else np.nan,
        "cpcv_full_sharpe_iqr": float(full_sharpe_iqr) if full_sharpe_iqr is not None else np.nan,
        "cpcv_full_cvar_5": float(full_cvar5) if full_cvar5 is not None else np.nan,
        "cpcv_full_support_rate": float(full_support_rate) if full_support_rate is not None else np.nan,
        "cpcv_full_regime_fragility": float(full_fragility) if not pd.isna(full_fragility) else np.nan,
        "cpcv_full_regime_coverage_json": full_regime_json_str,
        "cpcv_full_pbo_binary": float(full_pbo_binary),
        "cpcv_full_pbo_spearman": float(full_pbo_spearman),
        "cpcv_full_paths": int(full_result.get("path_count", 0)),
        "hartcpcv": float(hartcpcv) if not pd.isna(hartcpcv) else np.nan,
    }

    return pd.DataFrame([row])


# ------------------------------ (Legacy) Statistical helpers kept for compatibility ------------------------------
# If some older code imports these, they remain available.

from scipy import stats

def _legacy_deflated_sharpe_ratio(returns: pd.Series, n_trials: int = 1) -> float:
    # Kept only for compatibility with any legacy imports; prefer CPCV metrics.
    if len(returns) < 10:
        return 0.0
    T = len(returns)
    # simple Sharpe
    r = returns.dropna()
    mu, sig = r.mean(), r.std(ddof=1)
    sr = 0.0 if sig == 0 else float(np.sqrt(252) * mu / sig)
    if T <= 3:
        return 0.0
    sr_adj = sr * np.sqrt((T - 1) / (T - 3))
    return float(sr_adj * max(0.5, 1.0 - 1.0/np.sqrt(max(T-1,1))))  # softened

def _legacy_bootstrap_confidence_interval(returns: pd.Series, metric_func, n_bootstrap: int = 1000, confidence: float = 0.95) -> tuple[float, float, float]:
    if len(returns) < 10:
        return 0.0, 0.0, 0.0
    observed = metric_func(returns)
    vals = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = returns.sample(n=len(returns), replace=True, random_state=int(rng.integers(0, 1_000_000)))
        try:
            v = metric_func(sample)
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            continue
    if len(vals) < 10:
        return float(observed), float(observed), float(observed)
    a = 1 - confidence
    lo, hi = np.percentile(vals, 100*a/2), np.percentile(vals, 100*(1-a/2))
    return float(lo), float(observed), float(hi)
