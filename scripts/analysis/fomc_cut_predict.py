from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path

# Ensure repo root is importable when running file directly
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from alpha_discovery.config import settings  # type: ignore
    from alpha_discovery.features.registry import build_feature_matrix
except Exception:
    @dataclass
    class _EventsCfg:
        file_path: str = str(_REPO_ROOT / "data_store" / "processed" / "economic_releases.parquet")

    @dataclass
    class _DataCfg:
        parquet_file_path: str = str(_REPO_ROOT / "data_store" / "processed" / "bb_data.parquet")
        tradable_tickers: List[str] = field(default_factory=lambda: [
            'MSTR US Equity','SNOW US Equity','LLY US Equity','COIN US Equity',
            'QCOM US Equity','ULTA US Equity','CRM US Equity','AAPL US Equity',
            'AMZN US Equity','MSFT US Equity','QQQ US Equity','SPY US Equity'
        ])

    @dataclass
    class _FallbackSettings:
        events: _EventsCfg = field(default_factory=_EventsCfg)
        data: _DataCfg = field(default_factory=_DataCfg)

    settings = _FallbackSettings()

    def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
        # Minimal fallback: just return prices and daily pct change for tradables
        feats: Dict[str, pd.Series] = {}
        for t in settings.data.tradable_tickers:
            px = pd.to_numeric(df.get(f"{t}_PX_LAST", pd.Series(index=df.index)), errors="coerce")
            feats[f"{t}_px.mom_21"] = px.pct_change(21).shift(1)
        return pd.DataFrame(feats)

# Optional: XGBoost support
try:
    import xgboost as xgb  # type: ignore
    _XGB_OK = True
except Exception:
    _XGB_OK = False


# -----------------------------
# Data loading & helpers
# -----------------------------

def _load_calendar() -> pd.DataFrame:
    p = settings.events.file_path
    try:
        df = pd.read_parquet(p)
    except Exception:
        import os
        csv_guess = os.path.splitext(p)[0] + ".csv"
        df = pd.read_csv(csv_guess)
    # Normalize
    rename_map = {
        "release_datetime": "Date",
        "event_type": "Event",
        "country": "Country",
        "survey": "Survey",
        "actual": "Actual",
        "prior": "Prior",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    dt = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    try:
        dt = dt.dt.tz_convert("America/New_York").dt.tz_localize(None)
    except Exception:
        dt = pd.to_datetime(df["Date"], errors="coerce")
    df["ReleaseDate"] = dt.dt.floor("D")
    for c in ["Survey", "Actual", "Prior"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[df["Country"].astype(str).str.upper().eq("US")]


def _is_fomc_rate_decision(event: str) -> bool:
    return isinstance(event, str) and "FOMC Rate Decision" in event


def fomc_cut_days(cal: pd.DataFrame, expected_and_realized_only: bool = True) -> pd.DataFrame:
    df = cal[cal["Event"].apply(_is_fomc_rate_decision)].copy()
    upper = df[df["Event"].str.contains("Upper Bound", na=False)]
    base = upper if len(upper) else df
    base = base.dropna(subset=["ReleaseDate", "Prior"]).copy()
    base["delta_survey"] = base["Survey"] - base["Prior"]
    base["delta_actual"] = base["Actual"] - base["Prior"]
    base["surprise"] = base["Actual"] - base["Survey"]
    base["expected_cut"] = (base["delta_survey"] < 0).astype(int)
    base["realized_cut"] = (base["delta_actual"] < 0).astype(int)
    if expected_and_realized_only:
        base = base[(base["expected_cut"] == 1) & (base["realized_cut"] == 1)]
    cols = ["ReleaseDate", "Prior", "Survey", "Actual", "delta_survey", "delta_actual", "surprise", "expected_cut", "realized_cut"]
    return base.loc[:, cols].rename(columns={"ReleaseDate": "date"}).sort_values("date")


def _load_panel() -> pd.DataFrame:
    p = settings.data.parquet_file_path
    try:
        return pd.read_parquet(p)
    except Exception:
        import os
        csv_guess = os.path.splitext(p)[0] + ".csv"
        return pd.read_csv(csv_guess, parse_dates=[0], index_col=0)


def _resolve_base(panel: pd.DataFrame, user_ticker: str) -> Optional[str]:
    suffix = "_PX_LAST"
    for c in panel.columns:
        if c.endswith(suffix) and c.upper().startswith(user_ticker.upper() + " "):
            return c[: -len(suffix)]
    guess = f"{user_ticker} US Equity"
    if f"{guess}_PX_LAST" in panel.columns:
        return guess
    return None


def _add_bdays(d: pd.Timestamp, h: int) -> pd.Timestamp:
    rng = pd.bdate_range(d, periods=max(1, h + 1))
    return rng[-1]


def _oc_ret(panel: pd.DataFrame, base: str, d: pd.Timestamp, horizon_days: int = 0) -> Optional[float]:
    ocol = f"{base}_PX_OPEN"; ccol = f"{base}_PX_LAST"
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise ValueError("panel index must be DatetimeIndex")
    dd0 = pd.Timestamp(d)
    idx = panel.index
    if dd0 not in idx:
        prev = idx[idx <= dd0]
        if len(prev):
            dd0 = prev.max()
    # Exit day = business day shift forward
    ddh = _add_bdays(dd0, int(max(0, horizon_days)))
    if ddh not in idx:
        nxt = idx[idx >= ddh]
        if len(nxt):
            ddh = nxt.min()
    try:
        o = float(panel[ocol].loc[dd0])
        c = float(panel[ccol].loc[ddh])
    except Exception:
        return None
    if not np.isfinite(o) or not np.isfinite(c) or o == 0:
        return None
    return float(c / o - 1.0)


# -----------------------------
# Simple LOOCV linear classifier (2 features: event + registry)
# -----------------------------

def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


def _loocv_best_pair(X_evt: pd.DataFrame, X_reg: pd.DataFrame, y: pd.Series, top_k: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Grid search over pairs (one event feature, one registry feature).
    Fit linear model by least squares with LOOCV; predict yhat > 0.5.
    Return: top pairs table and per-date predictions for the best pair.
    """
    evt_cols = list(X_evt.columns)
    reg_cols = list(X_reg.columns)
    results = []
    best_preds = None
    best_key = None

    # Standardize helper
    def _stdz(a: np.ndarray) -> np.ndarray:
        m = np.nanmean(a, axis=0)
        s = np.nanstd(a, axis=0)
        s[s == 0] = 1.0
        return (a - m) / s

    for e in evt_cols:
        xe = X_evt[[e]].to_numpy(dtype=float)
        for r in reg_cols:
            xr = X_reg[[r]].to_numpy(dtype=float)
            X2 = np.concatenate([xe, xr], axis=1)
            # Replace NaNs with column means (computed per column)
            col_means = np.nanmean(X2, axis=0)
            inds = np.where(np.isnan(X2))
            if col_means.size > 0:
                X2[inds] = np.take(col_means, inds[1])
            X2 = _stdz(X2)
            yv = y.to_numpy(dtype=float)
            n = len(yv)
            preds = np.zeros(n, dtype=float)
            correct = np.zeros(n, dtype=bool)
            for i in range(n):
                mask = np.ones(n, dtype=bool); mask[i] = False
                if mask.sum() < 2:
                    continue
                Xtr = X2[mask]
                ytr = yv[mask]
                # Add bias
                Xtr_b = np.hstack([np.ones((Xtr.shape[0], 1)), Xtr])
                # Solve least squares
                try:
                    w, *_ = np.linalg.lstsq(Xtr_b, ytr, rcond=None)
                except Exception:
                    continue
                xte = np.hstack([1.0, X2[i]])
                score = float(np.dot(w, xte))
                p = _sigmoid(score)
                preds[i] = p
                correct[i] = (p > 0.5) == bool(yv[i] > 0.0)

            acc = float(correct.mean()) if correct.size else 0.0
            results.append({"evt": e, "reg": r, "acc": acc})
            if best_key is None or acc > results[best_key]["acc"]:
                best_key = len(results) - 1
                best_preds = preds.copy()

    res = pd.DataFrame(results).sort_values("acc", ascending=False).reset_index(drop=True)
    pred_df = pd.DataFrame({"y_true": y.values, "p_hat": best_preds, "pred": (np.array(best_preds) > 0.5).astype(int)}, index=y.index)
    return res.head(top_k), pred_df


def _mean_impute(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    if col_means.size > 0:
        X[inds] = np.take(col_means, inds[1])
    m = np.nanmean(X, axis=0)
    s = np.nanstd(X, axis=0); s[s == 0] = 1.0
    return X, m, s


def _loocv_xgb(X_df: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, float]:
    n = len(y)
    preds = np.zeros(n, dtype=float)
    correct = np.zeros(n, dtype=bool)
    X_raw = X_df.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        if mask.sum() < 3:
            preds[i] = 0.5; correct[i] = False; continue
        Xtr = X_raw[mask].copy(); ytr = yv[mask]
        Xtr, _, _ = _mean_impute(Xtr)
        clf = xgb.XGBClassifier(
            max_depth=3, n_estimators=500, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss",
            n_jobs=1, verbosity=0,
        )
        clf.fit(Xtr, ytr)
        xte = X_raw[i:i+1].copy()
        xte, _, _ = _mean_impute(xte)
        p = float(clf.predict_proba(xte)[0, 1])
        preds[i] = p
        correct[i] = (p > 0.5) == bool(yv[i] > 0.0)
    acc = float(correct.mean()) if n else 0.0
    return preds, acc


# -----------------------------
# Pipeline
# -----------------------------

def run_predict(ticker: str, only_expected_and_realized: bool = True, top_k_pairs: int = 5, horizon_days: int = 0,
                use_xgb: bool = False) -> Dict[str, object]:
    cal = _load_calendar()
    panel = _load_panel()
    base = _resolve_base(panel, ticker)
    if base is None:
        raise ValueError(f"Could not resolve columns for {ticker}")

    cuts = fomc_cut_days(cal, expected_and_realized_only=only_expected_and_realized)
    if cuts.empty:
        return {"summary": pd.DataFrame(), "pairs": pd.DataFrame(), "preds": pd.DataFrame(), "diagnostics": pd.DataFrame()}

    # Labels and realized O→C returns
    rets = []
    y_rows = []
    for d in cuts["date"].values:
        r = _oc_ret(panel, base, pd.Timestamp(d), horizon_days=horizon_days)
        rets.append(r if r is not None else np.nan)
        y_rows.append(1 if (r is not None and r > 0) else 0)
    y = pd.Series(y_rows, index=cuts["date"].values, name="y")
    oc = pd.Series(rets, index=cuts["date"].values, name="oc_ret")

    # Event feature candidates
    X_evt = cuts.set_index("date")[ ["delta_survey", "delta_actual", "surprise", "expected_cut", "realized_cut"] ]

    # Registry/Core features (ALL features, not just one ticker prefix)
    X_full = build_feature_matrix(panel)
    X_reg_all = X_full.reindex(y.index)
    X_reg_all = X_reg_all.dropna(how="all", axis=1)
    if X_reg_all.shape[1] == 0:
        return {"summary": pd.DataFrame(), "pairs": pd.DataFrame(), "preds": pd.DataFrame(), "diagnostics": pd.DataFrame()}

    if use_xgb and _XGB_OK:
        preds_vec, acc = _loocv_xgb(X_reg_all, y)
        pairs = pd.DataFrame({"model": ["xgb_all_features"], "acc": [acc]})
        preds = pd.DataFrame({"y_true": y.values, "p_hat": preds_vec, "pred": (preds_vec > 0.5).astype(int)}, index=y.index)
    else:
        # For pair search, limit registry candidates to this ticker’s features
        pref = f"{base}_"
        cols = [c for c in X_reg_all.columns if c.startswith(pref)]
        X_reg = X_reg_all.loc[:, cols]
        if X_reg.shape[1] == 0:
            return {"summary": pd.DataFrame(), "pairs": pd.DataFrame(), "preds": pd.DataFrame(), "diagnostics": pd.DataFrame()}
        pairs, preds = _loocv_best_pair(X_evt, X_reg, y, top_k=top_k_pairs)
    preds = preds.join(oc, how="left")

    # Diagnostics for best pair
    def _boot_ci_mean(a: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> Tuple[float, float, float]:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (np.nan, np.nan, np.nan)
        mean = float(np.nanmean(a))
        if a.size < 3:
            return (mean, np.nan, np.nan)
        rng = np.random.default_rng(42)
        idx = rng.integers(0, a.size, size=(n_boot, a.size))
        boots = np.nanmean(a[idx], axis=1)
        lo = float(np.nanpercentile(boots, 100 * (alpha / 2)))
        hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
        return (mean, lo, hi)

    d_long = preds.loc[preds["pred"] == 1, "oc_ret"].to_numpy(dtype=float)
    d_short = preds.loc[preds["pred"] == 0, "oc_ret"].to_numpy(dtype=float)
    meanL, loL, hiL = _boot_ci_mean(d_long)
    meanS, loS, hiS = _boot_ci_mean(d_short)
    winL = float(np.nanmean((d_long > 0).astype(float))) if d_long.size else np.nan
    winS = float(np.nanmean((d_short < 0).astype(float))) if d_short.size else np.nan
    diag = pd.DataFrame({
        "bucket": ["pred_long", "pred_short"],
        "n": [int((preds["pred"] == 1).sum()), int((preds["pred"] == 0).sum())],
        "mean_oc": [meanL, meanS],
        "ci_lo": [loL, loS],
        "ci_hi": [hiL, hiS],
        "hit_rate": [winL, winS],
        "avg_p_hat": [float(preds.loc[preds["pred"] == 1, "p_hat"].mean()) if (preds["pred"] == 1).any() else np.nan,
                       float(preds.loc[preds["pred"] == 0, "p_hat"].mean()) if (preds["pred"] == 0).any() else np.nan],
    })
    acc_best = float(pairs["acc"].iloc[0]) if len(pairs) else 0.0
    summary = pd.DataFrame({
        "ticker": [base],
        "n_obs": [int(len(y))],
        "best_acc": [acc_best],
        "best_evt": [pairs["evt"].iloc[0] if len(pairs) else None],
        "best_reg": [pairs["reg"].iloc[0] if len(pairs) else None],
        "pred_long_n": [int((preds["pred"] == 1).sum())],
        "pred_long_mean": [meanL],
        "pred_long_lo": [loL],
        "pred_long_hi": [hiL],
        "pred_long_hit": [winL],
        "pred_short_n": [int((preds["pred"] == 0).sum())],
        "pred_short_mean": [meanS],
        "pred_short_lo": [loS],
        "pred_short_hi": [hiS],
        "pred_short_hit": [winS],
    })
    preds = preds.reset_index().rename(columns={"index": "date"})
    return {"summary": summary, "pairs": pairs, "preds": preds, "diagnostics": diag}


def _fit_full_and_predict(X_evt: pd.DataFrame, X_reg: pd.DataFrame, y: pd.Series, evt_name: str, reg_name: str,
                          evt_row: pd.Series, reg_row: pd.Series) -> float:
    """Fit least-squares model on all historical data for the chosen pair, then score a new row."""
    xe = X_evt[[evt_name]].to_numpy(dtype=float)
    xr = X_reg[[reg_name]].to_numpy(dtype=float)
    X2 = np.concatenate([xe, xr], axis=1)
    # Mean-impute & standardize
    col_means = np.nanmean(X2, axis=0)
    inds = np.where(np.isnan(X2))
    if col_means.size > 0:
        X2[inds] = np.take(col_means, inds[1])
    m = np.nanmean(X2, axis=0); s = np.nanstd(X2, axis=0); s[s == 0] = 1.0
    X2 = (X2 - m) / s
    yv = y.to_numpy(dtype=float)
    # Fit with bias
    Xb = np.hstack([np.ones((X2.shape[0], 1)), X2])
    w, *_ = np.linalg.lstsq(Xb, yv, rcond=None)
    # Prepare the new row
    x_new = np.array([evt_row.get(evt_name, np.nan), reg_row.get(reg_name, np.nan)], dtype=float)
    # impute & standardize with training stats
    nan_mask = ~np.isfinite(x_new)
    if nan_mask.any():
        x_new[nan_mask] = col_means[nan_mask]
    x_new = (x_new - m) / s
    x_new_b = np.hstack([1.0, x_new])
    score = float(np.dot(w, x_new_b))
    return _sigmoid(score)


def _ols_pred_mean_ci(X2: np.ndarray, y: np.ndarray, x_new: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Fit OLS on (standardized) X2 -> y and return bootstrap CI for predicted mean at x_new.
    X2 and x_new should already be mean-imputed and standardized consistently.
    """
    # Add bias
    Xb = np.hstack([np.ones((X2.shape[0], 1)), X2])
    # Closed-form least squares
    w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    y_hat = float(np.dot(w, np.hstack([1.0, x_new])))

    # Parametric bootstrap over residuals
    resid = y - Xb.dot(w)
    if len(resid) < 3:
        return (y_hat, np.nan, np.nan)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        Xi = Xb[idx[i]]
        yi = y[idx[i]]
        wi, *_ = np.linalg.lstsq(Xi, yi, rcond=None)
        means[i] = float(np.dot(wi, np.hstack([1.0, x_new])))
    lo = float(np.nanpercentile(means, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(means, 100 * (1 - alpha / 2)))
    return (y_hat, lo, hi)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = (z * np.sqrt((p*(1-p) + z*z/(4*n)) / n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def predict_today(ticker: str, only_expected_and_realized: bool = True,
                  cost_bps: float = 5.0, min_ev_bps: float = 0.0,
                  horizon_days: int = 0, use_xgb: bool = False) -> Dict[str, object]:
    """Return a direction prediction for today using the best event+registry pair.
    Uses today's survey/prior to build the event row; actual/surprise may be NaN.
    Maps expected move to the historical bucket mean/CI for the predicted side.
    """
    res = run_predict(ticker, only_expected_and_realized=only_expected_and_realized, top_k_pairs=5, horizon_days=horizon_days, use_xgb=use_xgb)
    if res["pairs"].empty:
        return {"message": "No training data"}
    best_evt = res["pairs"]["evt"].iloc[0] if "evt" in res["pairs"].columns else None
    best_reg = res["pairs"]["reg"].iloc[0] if "reg" in res["pairs"].columns else None

    # Rebuild pieces used by training
    cal = _load_calendar(); panel = _load_panel()
    base = _resolve_base(panel, ticker)
    cuts = fomc_cut_days(cal, expected_and_realized_only=False)
    # Labels
    y_rows = []; rets = []
    for d in cuts["date"].values:
        r = _oc_ret(panel, base, pd.Timestamp(d), horizon_days=horizon_days)
        rets.append(r if r is not None else np.nan)
        y_rows.append(1 if (r is not None and r > 0) else 0)
    y = pd.Series(y_rows, index=cuts["date"].values, name="y")

    # Event matrix
    X_evt_all = cuts.set_index("date")[ ["delta_survey", "delta_actual", "surprise", "expected_cut", "realized_cut"] ]
    # Registry matrix (ALL features)
    X_full = build_feature_matrix(panel)
    X_reg_all = X_full.reindex(y.index).dropna(how="all", axis=1)
    if X_reg_all.shape[1] == 0:
        return {"message": "No registry features available"}

    # Build today's row
    today = pd.Timestamp.today().normalize()
    # Choose the latest FOMC row on/<= today
    today_evt = cal[cal["Event"].apply(_is_fomc_rate_decision)].copy()
    today_evt = today_evt.sort_values("ReleaseDate")
    row = today_evt.loc[today_evt["ReleaseDate"] <= today].tail(1)
    if row.empty:
        return {"message": "No FOMC row found <= today"}
    prior = float(row["Prior"].iloc[0]) if "Prior" in row.columns else np.nan
    survey = float(row["Survey"].iloc[0]) if "Survey" in row.columns else np.nan
    delta_survey = (survey - prior) if (np.isfinite(survey) and np.isfinite(prior)) else np.nan
    evt_row = pd.Series({
        "delta_survey": delta_survey,
        "delta_actual": np.nan,
        "surprise": np.nan,
        "expected_cut": float(delta_survey < 0) if np.isfinite(delta_survey) else np.nan,
        "realized_cut": np.nan,
    })

    # Registry feature row: take latest available date <= today
    ix = X_reg_all.index
    if len(ix) == 0:
        return {"message": "No feature index"}
    t_idx = ix[ix <= today]
    if len(t_idx) == 0:
        t_use = ix.max()
    else:
        t_use = t_idx.max()
    reg_row = X_reg_all.loc[t_use]

    if use_xgb and _XGB_OK:
        # Fit XGB on all data and score today
        X_raw = X_reg_all.to_numpy(dtype=float)
        X_raw, _, _ = _mean_impute(X_raw)
        clf = xgb.XGBClassifier(
            max_depth=3, n_estimators=500, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss",
            n_jobs=1, verbosity=0,
        )
        clf.fit(X_raw, y.to_numpy(dtype=float))
        ix = X_reg_all.index
        t_idx = ix[ix <= today]
        t_use = t_idx.max() if len(t_idx) else ix.max()
        reg_row = X_reg_all.loc[t_use]
        x_new = reg_row.to_numpy(dtype=float)[None, :]
        x_new, _, _ = _mean_impute(x_new)
        p_hat = float(clf.predict_proba(x_new)[0, 1])
    else:
        # Pair model
        if best_evt is None or best_reg is None:
            return {"message": "No pair available"}
        p_hat = _fit_full_and_predict(X_evt_all, X_reg_all, y, best_evt, best_reg, evt_row, reg_row)
    direction = "LONG" if p_hat > 0.5 else "SHORT"
    diag = res["diagnostics"]
    if direction == "LONG":
        expected = diag.loc[diag["bucket"] == "pred_long", ["mean_oc", "ci_lo", "ci_hi"]]
    else:
        expected = diag.loc[diag["bucket"] == "pred_short", ["mean_oc", "ci_lo", "ci_hi"]]
    exp_vals = expected.iloc[0].to_dict() if not expected.empty else {"mean_oc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    # EV calculation using bucket mean (trade-direction signed) and cost
    mean_ret = float(exp_vals.get("mean_oc", np.nan))
    ev_ret = mean_ret if direction == "LONG" else (-mean_ret)
    ev_bps = float(ev_ret * 10000.0 - cost_bps)
    decision = "TRADE" if (ev_bps >= min_ev_bps) else "NO-TRADE"
    # Model-based CI for return using OLS on the chosen pair
    # Build training design (impute/standardize) and new row same way as in _fit_full_and_predict
    if (best_evt is not None) and (best_reg is not None):
        xe = X_evt_all[[best_evt]].to_numpy(dtype=float)
        xr = X_reg_all[[best_reg]].to_numpy(dtype=float)
        X2 = np.concatenate([xe, xr], axis=1)
    else:
        X2 = np.empty((0, 2))
    # Align continuous returns
    oc_series = []
    for d in X_evt_all.index:
        oc_series.append(_oc_ret(panel, base, pd.Timestamp(d), horizon_days=horizon_days))
    y_cont = np.array([np.nan if v is None else float(v) for v in oc_series], dtype=float)
    mask = np.isfinite(y_cont)
    if mask.sum() >= 3 and X2.shape[1] == 2:
        X2m = X2[mask]
        ym = y_cont[mask]
        # Mean-impute + standardize on training
        col_means = np.nanmean(X2m, axis=0)
        inds = np.where(np.isnan(X2m))
        if col_means.size > 0:
            X2m[inds] = np.take(col_means, inds[1])
        m = np.nanmean(X2m, axis=0); s = np.nanstd(X2m, axis=0); s[s == 0] = 1.0
        X2m = (X2m - m) / s
        x_new = np.array([evt_row.get(best_evt, np.nan), reg_row.get(best_reg, np.nan)], dtype=float)
        nan_mask = ~np.isfinite(x_new)
        if nan_mask.any():
            x_new[nan_mask] = col_means[nan_mask]
        x_new = (x_new - m) / s
        model_mean, model_lo, model_hi = _ols_pred_mean_ci(X2m, ym, x_new, n_boot=2000)
    else:
        model_mean = model_lo = model_hi = np.nan

    return {
        "ticker": base, "best_evt": best_evt, "best_reg": best_reg, "p_hat": p_hat, "direction": direction,
        **exp_vals, "ev_bps": ev_bps, "cost_bps": cost_bps, "decision": decision,
        "model_mean": model_mean, "model_lo": model_lo, "model_hi": model_hi
    }


def scan_today_all(cost_bps: float = 5.0, min_ev_bps: float = 0.0, only_expected_and_realized: bool = True, horizon_days: int = 0) -> pd.DataFrame:
    rows = []
    for tk in getattr(settings.data, "tradable_tickers", []):
        short = tk.split(" ")[0]
        try:
            r = predict_today(short, only_expected_and_realized=only_expected_and_realized, cost_bps=cost_bps, min_ev_bps=min_ev_bps, horizon_days=horizon_days)
            if "p_hat" in r:
                rows.append({
                    "ticker": r.get("ticker"),
                    "p_hat": float(r.get("p_hat")),
                    "direction": r.get("direction"),
                    "ev_bps": float(r.get("ev_bps")),
                    "decision": r.get("decision"),
                    "expected_mean": float(r.get("mean_oc", np.nan)),
                    "ci_lo": float(r.get("ci_lo", np.nan)),
                    "ci_hi": float(r.get("ci_hi", np.nan)),
                })
        except Exception as e:
            print(f"[today scan warn] {short}: {e}")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["abs_p_hat"] = df["p_hat"].abs()
    return df.sort_values(["abs_p_hat", "ev_bps"], ascending=[False, False]).reset_index(drop=True)


def scan_all(top_k_pairs: int = 3, only_expected_and_realized: bool = True) -> pd.DataFrame:
    tickers = [t.split(" ")[0] for t in getattr(settings.data, "tradable_tickers", [])]
    out_rows = []
    for tk in tickers:
        try:
            res = run_predict(tk, only_expected_and_realized=only_expected_and_realized, top_k_pairs=top_k_pairs)
            if res["summary"].empty:
                continue
            out_rows.append(res["summary"]) 
        except Exception as e:
            print(f"[scan warn] {tk}: {e}")
    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True).sort_values(["best_acc", "n_obs"], ascending=[False, False])


def main():
    ap = argparse.ArgumentParser(description="Predict O→C direction on FOMC cut days with event+registry features")
    ap.add_argument("--ticker", default=None, help="Single ticker to evaluate (e.g., QQQ). If omitted with --scan, scan all.")
    ap.add_argument("--scan", action="store_true", help="Scan all tradable tickers and report best accuracy")
    ap.add_argument("--pairs", type=int, default=5, help="Top-K feature pairs to show for single-ticker mode")
    ap.add_argument("--out", default=None, help="CSV to write per-date predictions (single-ticker mode)")
    ap.add_argument("--diag_out", default=None, help="CSV to write diagnostics (single-ticker mode)")
    ap.add_argument("--summary_out", default=None, help="CSV to write scan summary table")
    ap.add_argument("--all", action="store_true", help="Include all FOMC decisions (not only expected+realized cuts)")
    ap.add_argument("--predict_today", action="store_true", help="Print a direction prediction for today with expected move CI")
    ap.add_argument("--cost_bps", type=float, default=5.0, help="Round-trip cost/slippage in bps for EV")
    ap.add_argument("--min_ev_bps", type=float, default=0.0, help="Minimum expected value (bps) required to trade")
    ap.add_argument("--rank_today", action="store_true", help="Rank all tradable tickers by |p_hat| for today and print top/bottom")
    ap.add_argument("--top", type=int, default=10, help="How many to show in rank_today mode")
    ap.add_argument("--horizon_days", type=int, default=0, help="Use open today to close at T+h business days (0,2,3,...) for labels and predictions")
    ap.add_argument("--xgb", action="store_true", help="Use XGBoost with all features instead of linear pair model")
    args = ap.parse_args()

    if args.scan:
        table = scan_all(top_k_pairs=3, only_expected_and_realized=(not args.all))
        print("\nScan results (best accuracy per ticker):\n")
        print(table.head(20))
        if args.summary_out:
            import os
            os.makedirs(Path(args.summary_out).parent, exist_ok=True)
            table.to_csv(args.summary_out, index=False)
            print(f"\nWrote scan summary → {args.summary_out}")
        return

    if args.rank_today:
        T = scan_today_all(cost_bps=float(args.cost_bps), min_ev_bps=float(args.min_ev_bps), only_expected_and_realized=(not args.all), horizon_days=int(args.horizon_days))
        if T.empty:
            print("No today predictions available.")
            return
        def _pct(x):
            return f"{x*100:.2f}%" if np.isfinite(x) else "nan"
        out = T.head(int(args.top)).copy()
        for c in ("expected_mean", "ci_lo", "ci_hi"):
            if c in out.columns:
                out[c] = out[c].apply(lambda v: _pct(v))
        print("\nTop by |p_hat| today:\n")
        print(out)
        if args.summary_out:
            out.to_csv(args.summary_out, index=False)
            print(f"Wrote rank_today summary → {args.summary_out}")
        return

    if not args.ticker:
        print("Please provide --ticker (or use --scan)")
        return
    res = run_predict(args.ticker, only_expected_and_realized=(not args.all), top_k_pairs=args.pairs, horizon_days=int(args.horizon_days), use_xgb=bool(args.xgb))
    print("\nBest pairs:\n")
    print(res["pairs"])  # type: ignore
    print("\nPer-date predictions:\n")
    print(res["preds"])  # type: ignore
    print("\nDiagnostics (by predicted direction):\n")
    print(res["diagnostics"])  # type: ignore
    if args.predict_today:
        today_res = predict_today(args.ticker, only_expected_and_realized=(not args.all), cost_bps=float(args.cost_bps), min_ev_bps=float(args.min_ev_bps), horizon_days=int(args.horizon_days), use_xgb=bool(args.xgb))
        if "direction" in today_res:
            def _pct(x):
                return f"{x*100:.2f}%" if x is not None and np.isfinite(x) else "nan"
            print("\nPrediction for today:")
            print({
                "ticker": today_res.get("ticker"),
                "pair": (today_res.get("best_evt"), today_res.get("best_reg")),
                "p_hat": round(float(today_res.get("p_hat", float("nan"))), 3),
                "direction": today_res.get("direction"),
                "expected_mean": _pct(today_res.get("mean_oc")),
                "expected_ci": (_pct(today_res.get("ci_lo")), _pct(today_res.get("ci_hi"))),
                "model_mean": _pct(today_res.get("model_mean")),
                "model_ci": (_pct(today_res.get("model_lo")), _pct(today_res.get("model_hi"))),
                "ev_bps": round(float(today_res.get("ev_bps", float("nan"))), 1),
                "cost_bps": float(today_res.get("cost_bps", float("nan"))),
                "decision": today_res.get("decision"),
            })
        else:
            print(f"\nPrediction for today unavailable: {today_res.get('message')}")
    if args.out and not res["preds"].empty:
        import os
        os.makedirs(Path(args.out).parent, exist_ok=True)
        P = res["preds"].copy()  # type: ignore
        if "oc_ret" in P.columns:
            P["oc_ret_pct"] = (P["oc_ret"].astype(float) * 100.0).round(2)
        P.to_csv(args.out, index=False)
        print(f"\nWrote predictions → {args.out}")
    if args.diag_out and not res["diagnostics"].empty:
        import os
        os.makedirs(Path(args.diag_out).parent, exist_ok=True)
        D = res["diagnostics"].copy()  # type: ignore
        for c in ("mean_oc", "ci_lo", "ci_hi"):
            if c in D.columns:
                D[c] = (D[c].astype(float) * 100.0).round(2)
        D.to_csv(args.diag_out, index=False)
        print(f"Wrote diagnostics → {args.diag_out}")


if __name__ == "__main__":
    main()


