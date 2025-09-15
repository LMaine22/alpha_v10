from __future__ import annotations
import os
import sys
import json
import argparse
import logging
import traceback
import joblib
import numpy as np
import pandas as pd
from typing import List

from .config import PredictConfig
from .pipeline import build_features
from .report import build_outlook_table, save_outlook_csv

LOG_FMT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("predict.forecast")

def _load_models(hdir: str):
    clf = joblib.load(os.path.join(hdir, "clf_dir.pkl"))
    q10 = joblib.load(os.path.join(hdir, "reg_q10.pkl"))
    q50 = joblib.load(os.path.join(hdir, "reg_q50.pkl"))
    q90 = joblib.load(os.path.join(hdir, "reg_q90.pkl"))
    with open(os.path.join(hdir, "manifest.json"), "r") as f:
        manifest = json.load(f)
    return clf, q10, q50, q90, manifest

def forecast_today(cfg: PredictConfig) -> pd.DataFrame:
    logger.info("=" * 64)
    logger.info("PREDICTIVE MINI-ENGINE: FORECAST")
    logger.info("=" * 64)
    logger.info(f"Using feature builder: {cfg.feature_builder_path}")
    feat = build_features(cfg)

    # Latest date per ticker
    latest_date = feat[cfg.pred_date_col].max()
    logger.info(f"Latest feature date detected: {latest_date}")
    live = feat[feat[cfg.pred_date_col] == latest_date].copy()
    logger.info(f"Live frame rows: {len(live):,} | cols: {live.shape[1]}")

    outputs = []
    for H in cfg.horizons:
        hdir = os.path.join(cfg.outdir, "models", f"h{H}")
        if not os.path.exists(hdir):
            raise RuntimeError(f"Trained model directory not found: {hdir}. Did you run train first?")
        logger.info(f"Loading models for H={H}d from {hdir}")
        clf, q10, q50, q90, man = _load_models(hdir)

        # Align columns exactly to training features
        X_cols: List[str] = man["feature_columns"]
        missing = [c for c in X_cols if c not in live.columns]
        if missing:
            raise RuntimeError(f"Live features missing training columns: {missing[:10]} (and possibly more).")
        X_live = live[X_cols].values

        logger.info(f"Scoring horizon H={H}d …")
        dir_prob = clf.predict_proba(X_live)[:, 1]
        ret_q10 = q10.predict(X_live)
        ret_q50 = q50.predict(X_live)
        ret_q90 = q90.predict(X_live)

        df_h = pd.DataFrame({
            cfg.pred_date_col: live[cfg.pred_date_col].values,
            cfg.ticker_col: live[cfg.ticker_col].values,
            "horizon_days": H,
            "dir_prob": dir_prob,
            "ret_q10": ret_q10,
            "ret_q50": ret_q50,
            "ret_q90": ret_q90,
        })
        outputs.append(df_h)

    full = pd.concat(outputs, axis=0, ignore_index=True)
    logger.info(f"Forecast rows generated: {len(full):,}")
    return full

def main():
    parser = argparse.ArgumentParser(description="Forecast latest outlooks with trained models.")
    parser.add_argument("--top", type=int, default=None, help="Top-N per horizon (overrides config).")
    parser.add_argument("--min-prob", type=float, default=None, help="Min directional prob (overrides config).")
    args = parser.parse_args()

    try:
        cfg = PredictConfig()
        df = forecast_today(cfg)
        # Build human-readable table
        top_n = args.top if args.top is not None else cfg.top_n_bullish
        prob_min = args.min_prob if args.min_prob is not None else cfg.confidence_min
        table = build_outlook_table(df, cfg, top_n=top_n, prob_min=prob_min)

        # Save CSV and echo path
        csv_path = save_outlook_csv(table, cfg, fname="latest_outlook.csv")
        logger.info("-" * 64)
        logger.info("Daily Outlook (top selections):")
        if len(table) == 0:
            logger.info("(No rows passed the confidence filter. Try lowering --min-prob.)")
        else:
            logger.info("\n" + table.to_string(index=False))
        logger.info("-" * 64)
        logger.info(f"Saved CSV: {csv_path}")
        logger.info("=" * 64)
    except Exception as e:
        logger.error("Forecast failed with an exception.")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
