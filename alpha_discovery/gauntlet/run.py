# alpha_discovery/gauntlet/run.py
"""
Gauntlet 2.0 - Main Runner

Orchestrates the 5-stage candidate evaluation pipeline:
1. Health & Sanity
2. WF Profitability  
3. CPCV Robustness
4. Portfolio Fit
5. Final Decision
"""
from __future__ import annotations

import os
import json
import time
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from ..config import Settings, gauntlet_cfg
from .io import find_latest_run_dir, read_global_artifacts, read_oos_fold_ledgers
from .stage1_health import run_stage1_health_check
from .stage2_profitability import run_stage2_profitability_wf, run_stage2_profitability
from .stage3_robustness import run_stage3_cpcv
from .stage4_portfolio import run_stage4_portfolio  
from .stage5_decision import run_stage5_final


def _safe_bool(df: Optional[pd.DataFrame], col: str) -> bool:
    """Safely extract boolean from DataFrame column."""
    if df is None or df.empty or col not in df.columns:
        return False
    val = df[col].iloc[0]
    return bool(val) if pd.notna(val) else False


def _write_run_manifest(
    run_dir: str,
    config: Dict[str, Any],
    start_time: float,
    stage_results: Dict[str, Optional[pd.DataFrame]],
    artifacts: Dict[str, str],
    smoke_mode: bool
) -> str:
    """Write run manifest for reproducibility."""
    gauntlet_dir = os.path.join(run_dir, "gauntlet")
    os.makedirs(gauntlet_dir, exist_ok=True)
    
    manifest = {
        "version": "2.0",
        "started_at": start_time,
        "ended_at": time.time(),
        "config": config,
        "stages": {},
        "smoke_mode": bool(smoke_mode),
        "artifacts": artifacts,
    }
    
    for stage_num in range(1, 6):
        stage_key = f"stage{stage_num}"
        df = stage_results.get(stage_key)
        pass_col = f"pass_stage{stage_num}"
        manifest["stages"][stage_key] = {
            "completed": df is not None and not df.empty,
            "passed": _safe_bool(df, pass_col),
            "reject_code": df[df.columns[df.columns.str.contains("reject_code")][0]].iloc[0] if df is not None and not df.empty and any(df.columns.str.contains("reject_code")) else None
        }
    
    manifest_path = os.path.join(gauntlet_dir, "run.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest_path


def _write_gauntlet_outputs(
    run_dir: str,
    stage_results: Dict[str, Optional[pd.DataFrame]],
    diagnostics: bool = True
) -> Dict[str, str]:
    """Write standardized gauntlet outputs."""
    gauntlet_dir = os.path.join(run_dir, "gauntlet")
    os.makedirs(gauntlet_dir, exist_ok=True)
    
    output_paths = {}
    
    # Main results file
    main_results = pd.DataFrame()
    for stage_num in range(1, 6):
        stage_key = f"stage{stage_num}"
        df = stage_results.get(stage_key)
        if df is not None and not df.empty:
            pass_col = f"pass_stage{stage_num}"
            core_cols = ["setup_id", "rank"]
            keep_plain = core_cols + [pass_col]

            plain_cols = [c for c in df.columns if c in keep_plain]
            diag_cols = [c for c in df.columns if c not in keep_plain + ["reject_code", "reason"]]

            df_prefixed = df[plain_cols].copy()
            if "reject_code" in df.columns:
                df_prefixed[f"s{stage_num}_reject_code"] = df["reject_code"]
            if "reason" in df.columns:
                df_prefixed[f"s{stage_num}_reason"] = df["reason"]
            for col in diag_cols:
                df_prefixed[f"s{stage_num}_{col}"] = df[col]
            
            if main_results.empty:
                main_results = df_prefixed
            else:
                main_results = pd.merge(main_results, df_prefixed, on=core_cols, how="outer")
    
    for stage_num in range(1, 6):
        pass_col = f"pass_stage{stage_num}"
        if pass_col not in main_results.columns:
            main_results[pass_col] = False

    if not main_results.empty:
        final_reject = pd.Series([None] * len(main_results), index=main_results.index, dtype=object)
        final_reason = pd.Series([None] * len(main_results), index=main_results.index, dtype=object)
        final_decision = pd.Series(["Retire"] * len(main_results), index=main_results.index, dtype=object)
        
        for stage_num in range(5, 0, -1):
            pass_col = f"pass_stage{stage_num}"
            rc_col = f"s{stage_num}_reject_code"
            reason_col = f"s{stage_num}_reason"
            if rc_col in main_results.columns:
                mask = final_reject.isna() & (~main_results[rc_col].isna()) & (~main_results[pass_col].astype(bool))
                final_reject.loc[mask] = main_results.loc[mask, rc_col]
                final_reason.loc[mask] = main_results.loc[mask, reason_col]
        
        # Set final decision based on stage 5 pass status and promotion score
        if "pass_stage5" in main_results.columns:
            stage5_passed = main_results["pass_stage5"].astype(bool)
            
            # Get promotion scores if available
            promotion_scores = main_results.get("promotion_score", pd.Series([0.0] * len(main_results)))
            
            # Decision logic:
            # - Deploy: passed all stages + high promotion score (>= 0.6)
            # - Monitor: passed all stages + moderate promotion score (>= 0.3)
            # - Retire: failed any stage or low promotion score
            
            deploy_mask = stage5_passed & (promotion_scores >= 0.6)
            monitor_mask = stage5_passed & (promotion_scores >= 0.3) & (promotion_scores < 0.6)
            
            final_decision.loc[deploy_mask] = "Deploy"
            final_decision.loc[monitor_mask] = "Monitor"
            # Retire is already the default
        
        # Create passed_gauntlet boolean column
        passed_gauntlet = final_decision.isin(["Deploy", "Monitor"])
        
        # Fallback to final stage reason if still missing (e.g., all pass)
        if "s5_reason" in main_results.columns:
            final_reason = final_reason.fillna(main_results["s5_reason"])
        if "s5_reject_code" in main_results.columns:
            final_reject = final_reject.fillna(main_results["s5_reject_code"])

        main_results["reject_code"] = final_reject
        main_results["reason"] = final_reason
        main_results["final_decision"] = final_decision
        main_results["passed_gauntlet"] = passed_gauntlet

    if not main_results.empty:
        results_path = os.path.join(gauntlet_dir, "gauntlet_results.csv")
        main_results.to_csv(results_path, index=False)
        output_paths["gauntlet_results"] = results_path
    
    # Stage diagnostics
    if diagnostics:
        diag_rows = []
        for stage_num in range(1, 6):
            stage_key = f"stage{stage_num}"
            df = stage_results.get(stage_key)
            if df is not None and not df.empty:
                df_copy = df.copy()
                df_copy["stage"] = stage_num
                diag_rows.append(df_copy)
        
        if diag_rows:
            diagnostics_df = pd.concat(diag_rows, ignore_index=True)
            diag_path = os.path.join(gauntlet_dir, "stage_diagnostics.csv")
            diagnostics_df.to_csv(diag_path, index=False)
            output_paths["stage_diagnostics"] = diag_path
    
    # Placeholder files for compatibility
    placeholder_files = {
        "gauntlet_ledger": ["setup_id", "trigger_date", "exit_date", "pnl_pct", "realized_pnl", "contracts"],
        "open_positions": ["setup_id", "ticker", "direction", "entry_date", "days_open", "unrealized_pnl", "contracts"]
    }
    
    for filename, columns in placeholder_files.items():
        filepath = os.path.join(gauntlet_dir, f"{filename}.csv")
        if not os.path.exists(filepath):
            pd.DataFrame(columns=columns).to_csv(filepath, index=False)
        output_paths[filename] = filepath
    
    return output_paths


def _ensure_columns(df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in required:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _json_dump_or_empty(value: Any) -> str:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "{}"
        return json.dumps(value)
    except TypeError:
        return json.dumps(str(value))


def _finalize_run(
    run_dir: str,
    cfg: Dict[str, Any],
    start_time: float,
    stage_results: Dict[str, Optional[pd.DataFrame]],
    diagnostics: bool,
    smoke_mode: bool
) -> Dict[str, str]:
    output_paths = _write_gauntlet_outputs(run_dir, stage_results, diagnostics=diagnostics)
    gauntlet_dir = os.path.join(run_dir, "gauntlet")
    artifacts = {
        "results_csv": output_paths.get("gauntlet_results"),
        "diagnostics_csv": output_paths.get("stage_diagnostics"),
    }
    artifacts = {k: v for k, v in artifacts.items() if v}
    artifacts["run_json"] = os.path.join(gauntlet_dir, "run.json")
    manifest_path = _write_run_manifest(
        run_dir=run_dir,
        config=cfg,
        start_time=start_time,
        stage_results=stage_results,
        artifacts=artifacts,
        smoke_mode=smoke_mode,
    )
    output_paths["run_json"] = manifest_path
    return output_paths


def _make_smoke_stage3_result(stage2_df: pd.DataFrame) -> pd.DataFrame:
    setup_id = stage2_df["setup_id"].iloc[0] if "setup_id" in stage2_df.columns else "unknown"
    rank = stage2_df.get("rank", pd.Series([None])).iloc[0]
    base = {
        "setup_id": setup_id,
        "rank": rank,
        "pass_stage3": True,
        "reject_code": None,
        "reason": "smoke_mode_skip",
    }
    placeholders = {
        "cpcv_lite_median_dsr": np.nan,
        "cpcv_lite_sharpe_iqr": np.nan,
        "cpcv_lite_cvar_5": np.nan,
        "cpcv_lite_support_rate": np.nan,
        "cpcv_lite_regime_fragility": np.nan,
        "cpcv_lite_regime_coverage_json": json.dumps({}),
        "cpcv_lite_pbo_binary": np.nan,
        "cpcv_lite_pbo_spearman": np.nan,
        "cpcv_full_median_dsr": np.nan,
        "cpcv_full_sharpe_iqr": np.nan,
        "cpcv_full_cvar_5": np.nan,
        "cpcv_full_support_rate": np.nan,
        "cpcv_full_regime_fragility": np.nan,
        "cpcv_full_regime_coverage_json": json.dumps({}),
        "cpcv_full_pbo_binary": np.nan,
        "cpcv_full_pbo_spearman": np.nan,
        "hartcpcv": 0.0,
        "cpcv_lite_paths": 0,
        "cpcv_full_paths": 0,
    }
    base.update(placeholders)
    return pd.DataFrame([base])


def run_gauntlet(
    run_dir: Optional[str] = None,
    settings: Optional[Settings] = None,
    config: Optional[Dict[str, Any]] = None,
    # Data inputs for CPCV/Portfolio stages  
    master_df: Optional[pd.DataFrame] = None,
    signals_df: Optional[pd.DataFrame] = None,
    signals_metadata: Optional[List[Dict[str, Any]]] = None,
    # Portfolio fit inputs
    candidate_returns: Optional[pd.Series] = None,
    live_returns_dict: Optional[Dict[str, pd.Series]] = None,
    live_activation_calendar: Optional[set] = None,
    # Candidate specification
    candidate_spec: Optional[Dict[str, Any]] = None,
    # Options
    diagnostics: bool = True,
) -> Dict[str, str]:
    """
    Run Gauntlet 2.0 - 5-stage candidate evaluation pipeline.
    
    Returns:
        Dictionary with paths to output files:
        - gauntlet_results.csv: Main pass/fail results
        - stage_diagnostics.csv: Detailed stage breakdowns
        - gauntlet_ledger.csv: Trade history (placeholder for now)
        - open_positions.csv: Open positions (placeholder for now)
    """
    start_time = time.time()
    
    # Setup configuration
    from ..config import settings as default_settings
    actual_settings = settings or default_settings  
    cfg = gauntlet_cfg(actual_settings)
    if config:
        cfg.update(config)
    smoke_mode = bool(cfg.get("smoke_mode", False))
    
    # Resolve run directory
    if run_dir is None:
        run_dir = find_latest_run_dir()
    if not run_dir or not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Could not resolve valid run_dir: {run_dir}")
    
    print(f"[Gauntlet 2.0] Starting 5-stage evaluation pipeline")
    print(f"[Gauntlet 2.0] Run directory: {run_dir}")
    
    # Get candidate info from existing artifacts if not provided
    if candidate_spec is None:
        pareto_summary, _ = read_global_artifacts(run_dir)
        if not pareto_summary.empty:
            candidate_spec = {
                "id": pareto_summary["setup_id"].iloc[0] if "setup_id" in pareto_summary.columns else "unknown",
                "signal_ids": [],
                "direction": "long",
                "ticker": "SPY US Equity"
            }
        else:
            candidate_spec = {"id": "unknown", "signal_ids": [], "direction": "long", "ticker": "SPY US Equity"}
    
    stage_results = {}
    
    # ===== STAGE 1: Health & Sanity =====
    print("[Stage 1] Health & Sanity checks...")
    
    # Create basic candidate info for Stage 1
    fold_summary = pd.DataFrame([{
        "setup_id": candidate_spec.get("id", "unknown"),
        "rank": 1
    }])
    
    # Try to get ledger from existing artifacts
    _, pareto_ledger = read_global_artifacts(run_dir)
    fold_ledger = pareto_ledger if not pareto_ledger.empty else pd.DataFrame()
    
    stage1_result = run_stage1_health_check(
        run_dir=run_dir,
        fold_num=None,
        settings=actual_settings,
        config=cfg.get("stage1", {}),
        fold_summary=fold_summary,
        fold_ledger=fold_ledger,
        market_df=master_df
    )
    
    stage_results["stage1"] = stage1_result
    
    if not _safe_bool(stage1_result, "pass_stage1"):
        print(f"[Stage 1] FAILED: {stage1_result['reason'].iloc[0]}")
        return _finalize_run(run_dir, cfg, start_time, stage_results, diagnostics, smoke_mode)
    
    print("[Stage 1] PASSED")
    
    # ===== STAGE 2: WF Profitability =====
    print("[Stage 2] WF Profitability gates...")
    
    wf_ledgers = read_oos_fold_ledgers(run_dir)
    if wf_ledgers:
        stage2_result = run_stage2_profitability_wf(
            fold_ledgers=wf_ledgers,
            settings=actual_settings,
            config=cfg.get("stage2", {}),
            stage1_df=stage1_result
        )
    else:
        stage2_result = run_stage2_profitability(
            run_dir=run_dir,
            fold_num=0,
            settings=actual_settings,
            config=cfg.get("stage2", {}),
            stage1_df=stage1_result
        )

    stage_results["stage2"] = stage2_result
    
    if not _safe_bool(stage2_result, "pass_stage2"):
        print(f"[Stage 2] FAILED: {stage2_result['reason'].iloc[0]}")
        return _finalize_run(run_dir, cfg, start_time, stage_results, diagnostics, smoke_mode)
    
    print("[Stage 2] PASSED")
    
    # ===== STAGE 3: CPCV Robustness =====
    print("[Stage 3] CPCV Robustness (lite → full)...")
    
    if smoke_mode:
        stage3_result = _make_smoke_stage3_result(stage2_result)
    else:
        try:
            stage3_result = run_stage3_cpcv(
                candidate_spec=candidate_spec,
                settings=actual_settings,
                config=cfg.get("stage3", {}),
                stage2_df=stage2_result,
                master_df=master_df,
                signals_df=signals_df,
                signals_metadata=signals_metadata
            )
        except Exception as e:
            print(f"[Stage 3] Error: {e}")
            stage3_result = pd.DataFrame([{
                "setup_id": candidate_spec.get("id", "unknown"),
                "rank": 1,
                "pass_stage3": False,
                "reject_code": "S3_ERROR", 
                "reason": f"stage3_failed: {e}"
            }])
    
    stage_results["stage3"] = stage3_result
    
    if not _safe_bool(stage3_result, "pass_stage3"):
        print(f"[Stage 3] FAILED: {stage3_result['reason'].iloc[0]}")
        return _finalize_run(run_dir, cfg, start_time, stage_results, diagnostics, smoke_mode)
    
    print("[Stage 3] PASSED")
    
    # ===== STAGE 4: Portfolio Fit =====
    print("[Stage 4] Portfolio Fit (correlation, capacity, overlap)...")
    
    try:
        stage4_result = run_stage4_portfolio(
            settings=actual_settings,
            config=cfg.get("stage4", {}),
            stage3_df=stage3_result,
            candidate_returns=candidate_returns,
            live_returns_dict=live_returns_dict,
            market_df=master_df,
            candidate_ledger=fold_ledger,
            live_activation_calendar=live_activation_calendar
        )
    except Exception as e:
        print(f"[Stage 4] Error: {e}")
        stage4_result = pd.DataFrame([{
            "setup_id": candidate_spec.get("id", "unknown"),
            "rank": 1,
            "pass_stage4": False,
            "reject_code": "S4_ERROR",
            "reason": f"stage4_failed: {e}"
        }])
    
    stage_results["stage4"] = stage4_result
    
    if not _safe_bool(stage4_result, "pass_stage4"):
        print(f"[Stage 4] FAILED: {stage4_result['reason'].iloc[0]}")
        return _finalize_run(run_dir, cfg, start_time, stage_results, diagnostics, smoke_mode)
    
    print("[Stage 4] PASSED")
    
    # ===== STAGE 5: Final Decision =====
    print("[Stage 5] Final Decision...")
    
    try:
        stage5_result = run_stage5_final(
            stage1_result, stage2_result, stage3_result, stage4_result,
            config=cfg.get("stage5", {})
        )
    except Exception as e:
        print(f"[Stage 5] Error: {e}")
        stage5_result = pd.DataFrame([{
            "setup_id": candidate_spec.get("id", "unknown"),
            "rank": 1,
            "pass_stage5": False,
            "reject_code": "S5_ERROR",
            "reason": f"stage5_failed: {e}",
            "promotion_score": 0.0
        }])
    
    stage_results["stage5"] = stage5_result
    
    final_passed = _safe_bool(stage5_result, "pass_stage5")
    if final_passed:
        print("[Stage 5] PASSED - Candidate APPROVED for promotion!")
    else:
        print(f"[Stage 5] FAILED: {stage5_result['reason'].iloc[0]}")
    
    # Write final outputs
    output_paths = _finalize_run(run_dir, cfg, start_time, stage_results, diagnostics, smoke_mode)
    
    total_time = time.time() - start_time
    print(f"[Gauntlet 2.0] Complete! Total time: {total_time:.1f}s")
    print(f"[Gauntlet 2.0] Final result: {'APPROVED' if final_passed else 'REJECTED'}")
    
    return output_paths