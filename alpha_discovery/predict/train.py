from __future__ import annotations
import os
import json
import sys
import traceback
import argparse
import logging
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd

from .config import PredictConfig
from .pipeline import build_features, attach_targets, make_splits, feature_target_columns
from .models import make_classifier, make_regressor
from .metrics import mae, auc, brier, ece_bin, summarize_regression_scores

LOG_FMT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("predict.train")

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _print_header():
    logger.info("=" * 64)
    logger.info("PREDICTIVE MINI-ENGINE: TRAIN")
    logger.info("=" * 64)

def _print_footer(outdir: str):
    logger.info("-" * 64)
    logger.info("Training complete.")
    logger.info(f"Artifacts directory: {outdir}")
    logger.info(f"Manifests: {os.path.join(outdir, 'train_manifest.json')}")
    logger.info("=" * 64)

def train_all(cfg: PredictConfig, debug_rows: int = 0) -> None:
    _print_header()
    logger.info(f"Using feature builder: {cfg.feature_builder_path}")
    logger.info(f"Using split builder  : {cfg.split_builder_path}")
    logger.info(f"Horizons (days)      : {cfg.horizons}")
    logger.info(f"Output dir           : {cfg.outdir}")

    _ensure_dir(cfg.outdir)

    # ---- Build features
    logger.info("Building features…")
    feat, panel_data = build_features(cfg)
    logger.info(f"Features shape: {feat.shape} | Columns: {len(feat.columns)}")
    if debug_rows > 0:
        logger.info("Feature sample (head):")
        logger.info("\n" + feat.head(debug_rows).to_string())

    # ---- Targets
    logger.info("Attaching forward-return and direction targets…")
    data = attach_targets(feat, panel_data, cfg)
    logger.info(f"Data shape (with targets): {data.shape}")
    
    # Filter out rows with NaN targets
    valid_mask = ~(data[f"ret_fwd_{cfg.horizons[0]}d"].isna() | 
                   data[f"target_dir_{cfg.horizons[0]}d"].isna())
    data = data[valid_mask].reset_index(drop=True)
    logger.info(f"Data shape after filtering NaN targets: {data.shape}")

    # ---- Splits
    logger.info("Building walk-forward splits…")
    splits = make_splits(data, cfg)
    logger.info(f"Number of splits: {len(splits)}")
    for i, (tr, te) in enumerate(splits, 1):
        logger.info(f"  Split {i}: train={len(tr):,} rows | test={len(te):,} rows")

    # ---- Train per-horizon models
    for H in cfg.horizons:
        logger.info("-" * 64)
        logger.info(f"Training models for horizon H={H}d")
        hdir = os.path.join(cfg.outdir, "models", f"h{H}")
        _ensure_dir(hdir)

        X_cols, y_cols = feature_target_columns(data, cfg, H)
        X = data[X_cols].values
        y_dir = data[f"target_dir_{H}d"].values.astype(float)
        y_ret = data[f"ret_fwd_{H}d"].values.astype(float)

        # Pool train indices (calibration happens inside classifier)
        train_idx = np.concatenate([tr for tr, _ in splits])
        test_idx = np.concatenate([te for _, te in splits])

        # --- Direction classifier
        logger.info("Fitting direction classifier…")
        clf = make_classifier(cfg.model_configs["clf_dir"])
        clf.fit(X[train_idx], y_dir[train_idx])

        logger.info("Scoring pooled OOS (direction)…")
        proba = clf.predict_proba(X[test_idx])[:, 1]
        auc_oos = auc(y_dir[test_idx], proba)
        brier_oos = brier(y_dir[test_idx], proba)
        ece_oos = ece_bin(y_dir[test_idx], proba, n_bins=10)
        logger.info(f"[OOS Direction] AUC={auc_oos:.3f} | Brier={brier_oos:.4f} | ECE={ece_oos:.4f}")

        joblib.dump(clf, os.path.join(hdir, "clf_dir.pkl"))
        logger.info(f"Saved: {os.path.join(hdir, 'clf_dir.pkl')}")

        # --- Quantile regressors
        logger.info("Fitting quantile regressors (q10, q50, q90)…")
        q10 = make_regressor(cfg.model_configs["reg_q_low"])
        q50 = make_regressor(cfg.model_configs["reg_median"])
        q90 = make_regressor(cfg.model_configs["reg_q_high"])

        q10.fit(X[train_idx], y_ret[train_idx])
        q50.fit(X[train_idx], y_ret[train_idx])
        q90.fit(X[train_idx], y_ret[train_idx])

        q10_pred = q10.predict(X[test_idx])
        q50_pred = q50.predict(X[test_idx])
        q90_pred = q90.predict(X[test_idx])

        reg_scores = summarize_regression_scores(y_ret[test_idx], q10_pred, q50_pred, q90_pred)
        logger.info(f"[OOS Returns]  MAE={reg_scores['MAE']:.5f} | WIS={reg_scores['WIS']:.5f}")

        joblib.dump(q10, os.path.join(hdir, "reg_q10.pkl"))
        joblib.dump(q50, os.path.join(hdir, "reg_q50.pkl"))
        joblib.dump(q90, os.path.join(hdir, "reg_q90.pkl"))
        logger.info(f"Saved: {os.path.join(hdir, 'reg_q10.pkl')}")
        logger.info(f"Saved: {os.path.join(hdir, 'reg_q50.pkl')}")
        logger.info(f"Saved: {os.path.join(hdir, 'reg_q90.pkl')}")

        # Save manifest
        manifest = {
            "horizon_days": H,
            "n_features": len(X_cols),
            "feature_columns": X_cols,
            "dir_oos_auc": auc_oos,
            "dir_oos_brier": brier_oos,
            "dir_oos_ece": ece_oos,
            "reg_oos_mae": reg_scores["MAE"],
            "reg_oos_wis": reg_scores["WIS"],
        }
        with open(os.path.join(hdir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Saved: {os.path.join(hdir, 'manifest.json')}")

    # Save global manifest
    with open(os.path.join(cfg.outdir, "train_manifest.json"), "w") as f:
        json.dump({"horizons": cfg.horizons}, f, indent=2)

    _print_footer(cfg.outdir)

def main():
    parser = argparse.ArgumentParser(description="Train predictive mini-engine models.")
    parser.add_argument("--debug-rows", type=int, default=0, help="Print head() rows of the feature frame.")
    args = parser.parse_args()

    try:
        cfg = PredictConfig()
        train_all(cfg, debug_rows=args.debug_rows)
    except Exception as e:
        logger.error("Training failed with an exception.")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
