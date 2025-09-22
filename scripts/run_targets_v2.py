# scripts/run_targets_v2.py
from __future__ import annotations
import argparse

from alpha_discovery.targets_v2.io import read_ledger, ensure_outdir
from alpha_discovery.targets_v2.labeling import compute_aux_labels
from alpha_discovery.targets_v2.curves import (
    option_pnl_curve,
    spot_only_curve,
    synthetic_nf_curve,
)
from alpha_discovery.targets_v2.half_life import compute_alpha_half_life
from alpha_discovery.targets_v2.diagnostics import summarize

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True, help="Path to your trade ledger CSV/Parquet.")
    ap.add_argument("--run-dir", required=True, help="Directory for outputs (e.g., runs/your_run)")
    args = ap.parse_args()

    trades = read_ledger(args.ledger)
    outdir = ensure_outdir(args.run_dir)

    # Enrich & save
    trades = compute_aux_labels(trades)
    trades["alpha_half_life"] = compute_alpha_half_life(trades)
    trades.to_parquet(outdir / "trades_enriched.parquet", index=False)

    # Curves
    c_opt = option_pnl_curve(trades)
    c_spot = spot_only_curve(trades)
    c_nf = synthetic_nf_curve(trades)

    c_opt.to_parquet(outdir / "curves_option_pnl.parquet", index=False)
    c_spot.to_parquet(outdir / "curves_spot_only.parquet", index=False)
    c_nf.to_parquet(outdir / "curves_synthetic_nf.parquet", index=False)

    # Diagnostics
    diag = summarize(trades, c_opt, c_spot, c_nf)
    diag.to_parquet(outdir / "diagnostics_summary.parquet", index=False)

    # Quick CSV previews (optional eyeballing)
    trades.head(2000).to_csv(outdir / "trades_enriched_sample.csv", index=False)
    c_opt.head(2000).to_csv(outdir / "curves_option_pnl_sample.csv", index=False)
    c_spot.head(2000).to_csv(outdir / "curves_spot_only_sample.csv", index=False)
    c_nf.head(2000).to_csv(outdir / "curves_synthetic_nf_sample.csv", index=False)
    diag.to_csv(outdir / "diagnostics_summary.csv", index=False)

    print(f"[TargetsV2] Wrote artifacts to: {outdir}")

if __name__ == "__main__":
    main()
