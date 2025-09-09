import argparse
import glob
import os
import sys
from .run import run_gauntlet_strict_oos

def _pick_latest_run_dir(base="runs"):
    cands = [p for p in glob.glob(os.path.join(base, "*")) if os.path.isdir(p)]
    if not cands:
        raise SystemExit(f"No run directories found under '{base}/'.")
    cands.sort(key=os.path.getmtime, reverse=True)
    return cands[0]

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Strict-OOS Gauntlet on an existing run directory.")
    parser.add_argument("--run-dir", type=str, default=None, help="Path to a specific runs/<...> directory. If omitted, pick the newest.")
    parser.add_argument("--outdir", type=str, default=None, help="Optional override for output directory.")
    parser.add_argument("--diagnostic-replay", action="store_true", help="Also create diagnostic_replay placeholder outputs.")
    parser.add_argument("--stage1-recency-days", type=int, default=14, help="OOS-aware recency window (days) for Stage 1 compat.")
    parser.add_argument("--stage1-min-trades", type=int, default=5, help="Minimum OOS trades per setup for Stage 1 compat.")
    # Reserved flags for future: --splits-json, --config, etc.
    args = parser.parse_args(argv)

    run_dir = args.run_dir or _pick_latest_run_dir()
    print(f"[Strict-OOS] Using run_dir: {run_dir}")
    res = run_gauntlet_strict_oos(
        run_dir=run_dir,
        splits=None,
        outdir=args.outdir,
        stage1_recency_days=args.stage1_recency_days,
        stage1_min_trades=args.stage1_min_trades
    )

    print("[Strict-OOS] Results:", res)
    print("[Strict-OOS] Output dir:", res.get("outdir"))

    if args.diagnostic_replay:
        try:
            from . import cli_diagnostic_replay as diag_cli
            diag_cli.main(["--run-dir", run_dir])
        except Exception as e:
            print("[Strict-OOS] Warning: diagnostic replay hook failed ->", e)

if __name__ == "__main__":
    main()
