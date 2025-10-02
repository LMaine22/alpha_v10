# alpha_discovery/gauntlet/stage4_portfolio.py
"""
Gauntlet 2.0 — Stage 4: Portfolio Fit (Capacity, Correlation, Overlap)

Purpose
-------
Ensure candidates that passed CPCV robustness actually fit the live portfolio:
    1) Capacity/slippage: haircut expected performance if venue is thin
    2) Correlation crowding: avoid duplicates of existing risk
    3) Activation overlap: avoid stacking fills on the same crowded days

Inputs
------
- stage3_df: single-row DF from Stage 3. Expected fields (best effort):
    * 'cpcv_full_median_dsr' (preferred) or 'cpcv_lite_median_dsr'
    * optionally: 'regime_summary_full'/'regime_summary_lite' (not required here)
- candidate_returns: pd.Series of OOS candidate returns (daily or bar), index datetime (optional but recommended)
- live_returns_dict: dict[str, pd.Series] of live strategies' OOS returns (aligned freq/index preferred)
- market_df: daily market features for the same asset/period (optional; capacity proxies live here)
    * expects some of: 'TOT_OPT_VOLUME_CUR_DAY', 'OPEN_INT_TOTAL_CALL', 'OPEN_INT_TOTAL_PUT', 'PX_VOLUME'
- candidate_ledger: trade-level DF for the candidate, to infer activation dates (optional)
- live_activation_calendar: set[datetime.date] of days where live portfolio is already active/crowded (optional)

Config keys (defaults)
----------------------
s4_corr_abs_max: float = 0.35      # avg |rho| threshold vs live book
s4_capacity_dsr_min: float = 0.20  # minimum DSR after capacity haircut
s4_overlap_max: float = 0.35       # allowed share of activation overlap with live calendar
s4_min_observations: int = 20

# capacity proxy floors (very rough, tune to your venue)
s4_min_tot_opt_volume: int = 1000
s4_min_open_interest_sum: int = 5000
s4_min_px_volume: float = 1_000_000.0

# weights for promotion score (diagnostic only; we still gate hard)
s4_w_dsr: float = 0.6
s4_w_uncorr: float = 0.4

Outputs
-------
Single-row DataFrame:
    pass_stage4, reject_code, reason,
    dsr_input, dsr_capacity_haircut, dsr_after_capacity,
    avg_abs_corr, n_live_series, overlap_share, capacity_score,
    plus raw pieces for auditability.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple, Set
import numpy as np
import pandas as pd


# ------------------------------ Helpers ------------------------------

def _safe_series(x) -> pd.Series:
    if x is None:
        return pd.Series(dtype=float)
    s = pd.Series(x) if not isinstance(x, pd.Series) else x
    s = pd.to_numeric(s, errors="coerce")
    return s.dropna()

def _align_returns(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    if a is None or b is None or a.empty or b.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    a, b = a.copy(), b.copy()
    a.index = pd.to_datetime(a.index); b.index = pd.to_datetime(b.index)
    idx = a.index.intersection(b.index)
    return _safe_series(a.loc[idx]), _safe_series(b.loc[idx])

def _avg_abs_corr(candidate: Optional[pd.Series], live_dict: Optional[Dict[str, pd.Series]]) -> Tuple[float, int, Dict[str, float]]:
    if candidate is None or candidate.empty or not live_dict:
        return np.nan, 0, {}
    cors = {}
    for name, s in live_dict.items():
        a, b = _align_returns(candidate, s)
        if len(a) >= 10 and len(b) >= 10:
            c = float(np.corrcoef(a, b)[0,1]) if np.isfinite(np.corrcoef(a, b)[0,1]) else np.nan
            if np.isfinite(c):
                cors[name] = c
    if not cors:
        return np.nan, 0, {}
    vals = np.abs(np.array(list(cors.values())))
    return float(vals.mean()), int(len(cors)), cors

def _capacity_score(market_df: Optional[pd.DataFrame],
                    min_tot_opt_volume: int,
                    min_open_interest_sum: int,
                    min_px_volume: float) -> Tuple[float, Dict[str, float]]:
    """
    Returns (capacity_score in [0,1], pieces).
    Heuristic: score is mean of three per-feature scores (clipped 0..1).
    """
    pieces = dict(vol_share=np.nan, oi_share=np.nan, pxv_share=np.nan)
    if market_df is None or market_df.empty:
        return np.nan, pieces
    df = market_df.copy()
    out = []

    # TOT_OPT_VOLUME_CUR_DAY
    if "TOT_OPT_VOLUME_CUR_DAY" in df.columns:
        ok = (pd.to_numeric(df["TOT_OPT_VOLUME_CUR_DAY"], errors="coerce") >= min_tot_opt_volume)
        pieces["vol_share"] = float(ok.mean())
        out.append(pieces["vol_share"])
    # OPEN_INT_TOTAL_CALL + PUT
    oi_parts = []
    for c in ["OPEN_INT_TOTAL_CALL", "OPEN_INT_TOTAL_PUT"]:
        if c in df.columns:
            oi_parts.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0))
    if oi_parts:
        oi_sum = sum(oi_parts)
        ok = (oi_sum >= min_open_interest_sum)
        pieces["oi_share"] = float(ok.mean())
        out.append(pieces["oi_share"])
    # PX_VOLUME
    if "PX_VOLUME" in df.columns:
        ok = (pd.to_numeric(df["PX_VOLUME"], errors="coerce") >= min_px_volume)
        pieces["pxv_share"] = float(ok.mean())
        out.append(pieces["pxv_share"])

    if not out:
        return np.nan, pieces

    score = float(np.clip(np.mean(out), 0.0, 1.0))
    return score, pieces

def _capacity_haircut(dsr_input: float, capacity_score: float) -> float:
    """
    Convert a 0..1 capacity_score to a DSR haircut.
    Example: linear: dsr_after = dsr_input * (0.5 + 0.5*capacity_score)
    => at score=0 → 0.5x; score=1 → 1.0x. Tune if you prefer stronger cuts.
    """
    if np.isnan(capacity_score):
        return float(dsr_input)
    factor = 0.5 + 0.5 * float(capacity_score)
    return float(dsr_input * factor)

def _activation_overlap(candidate_ledger: Optional[pd.DataFrame],
                        live_activation_calendar: Optional[Set[pd.Timestamp]]) -> Tuple[float, int, int]:
    """
    Overlap share = |candidate_active_days ∩ live_active_days| / |candidate_active_days|
    Uses trigger_date or entry_date as activation day.
    """
    if candidate_ledger is None or candidate_ledger.empty or not live_activation_calendar:
        return np.nan, 0, 0
    led = candidate_ledger.copy()
    dt_col = None
    for c in ("trigger_date", "entry_date", "exit_date"):
        if c in led.columns:
            dt_col = c; break
    if dt_col is None:
        return np.nan, 0, 0
    days = pd.to_datetime(led[dt_col], errors="coerce").dt.normalize().dropna().unique()
    cand_days = set(pd.to_datetime(pd.Index(days)).date)
    live_days = set(pd.to_datetime(pd.Index(list(live_activation_calendar))).date)
    if not cand_days:
        return np.nan, 0, 0
    overlap = len(cand_days & live_days)
    return float(overlap / max(1, len(cand_days))), int(overlap), int(len(cand_days))


# ------------------------------ Main ------------------------------

def run_stage4_portfolio(
    settings,
    config: Optional[Dict[str, Any]] = None,
    stage3_df: Optional[pd.DataFrame] = None,
    candidate_returns: Optional[pd.Series] = None,
    live_returns_dict: Optional[Dict[str, pd.Series]] = None,
    market_df: Optional[pd.DataFrame] = None,
    candidate_ledger: Optional[pd.DataFrame] = None,
    live_activation_calendar: Optional[set] = None,
) -> pd.DataFrame:
    """
    Stage 4: Portfolio Fit gates + diagnostics.

    - Requires Stage 3 results.
    - If returns/market_df are missing, related checks are skipped (not auto-fail).
    """
    cfg = dict(config or {})
    corr_abs_max       = float(cfg.get("s4_corr_abs_max", 0.35))
    capacity_dsr_min   = float(cfg.get("s4_capacity_dsr_min", 0.20))
    overlap_max        = float(cfg.get("s4_overlap_max", 0.35))
    min_obs            = int(cfg.get("s4_min_observations", 20))
    # capacity floors
    min_tot_opt_volume = int(cfg.get("s4_min_tot_opt_volume", 1000))
    min_oi_sum         = int(cfg.get("s4_min_open_interest_sum", 5000))
    min_px_volume      = float(cfg.get("s4_min_px_volume", 1_000_000.0))
    # promotion score weights (diagnostic)
    w_dsr              = float(cfg.get("s4_w_dsr", 0.6))
    w_uncorr           = float(cfg.get("s4_w_uncorr", 0.4))

    if stage3_df is None or stage3_df.empty:
        return pd.DataFrame([{
            "setup_id": None, "rank": None, "pass_stage4": False,
            "reject_code": "S4_INPUT_MISSING", "reason": "missing_stage3",
        }])

    setup_id = stage3_df["setup_id"].iloc[0]
    rank     = stage3_df.get("rank", pd.Series([None])).iloc[0]

    # DSR input from Stage 3 (prefer full, fall back to lite)
    dsr_input = np.nan
    for c in ("cpcv_full_median_dsr", "cpcv_lite_median_dsr"):
        if c in stage3_df.columns and pd.notnull(stage3_df[c].iloc[0]):
            dsr_input = float(stage3_df[c].iloc[0]); break

    # Capacity haircut
    cap_score, cap_pieces = _capacity_score(market_df, min_tot_opt_volume, min_oi_sum, min_px_volume)
    dsr_after_capacity = _capacity_haircut(dsr_input, cap_score) if pd.notnull(dsr_input) else np.nan

    # Correlation crowding (avg |rho|)
    avg_abs_corr, n_live, corr_map = _avg_abs_corr(candidate_returns, live_returns_dict)

    # Activation overlap
    overlap_share, overlap_n, cand_active_n = _activation_overlap(candidate_ledger, live_activation_calendar)

    # ---------------- Gates ----------------
    failures = []
    reject_code = None

    def _gate(ok: bool, code: str, msg: str):
        nonlocal reject_code
        if not ok:
            failures.append(msg)
            if reject_code is None:
                reject_code = code

    # DSR after capacity
    if not np.isnan(dsr_after_capacity):
        _gate(dsr_after_capacity >= capacity_dsr_min, "S4_CAPACITY",
              f"DSR_after_capacity={dsr_after_capacity:.2f}<{capacity_dsr_min:.2f}")

    # Correlation crowding
    if not np.isnan(avg_abs_corr):
        _gate(avg_abs_corr <= corr_abs_max, "S4_CORR",
              f"avg_abs_corr={avg_abs_corr:.2f}>{corr_abs_max:.2f}")

    # Overlap risk
    if not np.isnan(overlap_share):
        _gate(overlap_share <= overlap_max, "S4_OVERLAP",
              f"overlap_share={overlap_share:.2f}>{overlap_max:.2f}")

    passed = len(failures) == 0
    reason = "ok" if passed else ";".join(failures) or "failed"

    # Promotion score (diagnostic only)
    # higher is better: combine capacity-adjusted DSR and (1-avg|corr|)
    uncorr = (1.0 - avg_abs_corr) if np.isfinite(avg_abs_corr) else np.nan
    promo_score = np.nan
    if np.isfinite(dsr_after_capacity) and np.isfinite(uncorr):
        # min-max-lite normalization not applied here; prefer comparable scales
        promo_score = float(w_dsr * dsr_after_capacity + w_uncorr * uncorr)

    out = {
        "setup_id": setup_id,
        "rank": rank,
        "pass_stage4": bool(passed),
        "reject_code": None if passed else (reject_code or "S4_FAIL"),
        "reason": reason,

        # Inputs/diagnostics
        "dsr_input": float(dsr_input) if pd.notnull(dsr_input) else np.nan,
        "capacity_score": float(cap_score) if pd.notnull(cap_score) else np.nan,
        "dsr_after_capacity": float(dsr_after_capacity) if pd.notnull(dsr_after_capacity) else np.nan,

        "avg_abs_corr": float(avg_abs_corr) if pd.notnull(avg_abs_corr) else np.nan,
        "n_live_series": int(n_live),
        "corr_map": corr_map,  # dict[str,float] for audit

        "overlap_share": float(overlap_share) if pd.notnull(overlap_share) else np.nan,
        "overlap_n": int(overlap_n),
        "candidate_active_days": int(cand_active_n),

        # Floors used (for the run manifest / audit)
        "min_tot_opt_volume": int(min_tot_opt_volume),
        "min_open_interest_sum": int(min_oi_sum),
        "min_px_volume": float(min_px_volume),

        # Gates for traceability
        "s4_corr_abs_max": float(corr_abs_max),
        "s4_capacity_dsr_min": float(capacity_dsr_min),
        "s4_overlap_max": float(overlap_max),

        # Promotion (diagnostic)
        "promotion_score": float(promo_score) if pd.notnull(promo_score) else np.nan,
        "w_dsr": float(w_dsr),
        "w_uncorr": float(w_uncorr),
    }
    return pd.DataFrame([out])
