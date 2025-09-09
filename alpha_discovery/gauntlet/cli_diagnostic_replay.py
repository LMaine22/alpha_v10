import argparse
import glob
import os
import sys
import pandas as pd

from .diagnostic_replay import build_diagnostic_replay, summarize_diagnostic_replay
from .portfolio_diag import simulate_portfolio
from .run import _ensure_dir  # reuse helper

def _pick_latest_run_dir(base="runs"):
    cands = [p for p in glob.glob(os.path.join(base, "*")) if os.path.isdir(p)]
    if not cands:
        raise SystemExit(f"No run directories found under '{base}/'.")
    cands.sort(key=os.path.getmtime, reverse=True)
    return cands[0]

def main(argv=None):
    parser = argparse.ArgumentParser(description="Diagnostic Replay (non-gating). Replays IS+OOS with time-of-knowledge tagging.")
    parser.add_argument("--run-dir", type=str, default=None, help="Path to a specific runs/<...> directory. If omitted, pick the newest.")
    parser.add_argument("--all-setups", action="store_true", help="Replay all setups present in IS+OOS (default: survivors-only).")
    parser.add_argument("--since-knowledge", action="store_true", help="Drop trades flagged as uses_pre_knowledge.")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Maximum concurrent setups allowed.")
    parser.add_argument("--position-size", type=float, default=1000.0, help="Nominal dollars per trade if returns (not pnl) are used.")
    parser.add_argument("--starting-capital", type=float, default=100000.0, help="Starting capital for equity curve.")
    args = parser.parse_args(argv)

    run_dir = args.run_dir or _pick_latest_run_dir()
    out_base = os.path.join(run_dir, 'gauntlet', 'diagnostic_replay')
    os.makedirs(out_base, exist_ok=True)

    print("[Diagnostic] Building replay from:", run_dir)
    df = build_diagnostic_replay(run_dir=run_dir, splits=None, survivors_only=not args.all_setups)

    ledger_path = os.path.join(out_base, 'diag_replay_ledger.csv')
    df.to_csv(ledger_path, index=False)
    print("[Diagnostic] Wrote:", ledger_path, "rows:", len(df))

    summary = summarize_diagnostic_replay(df)
    summary_path = os.path.join(out_base, 'diag_replay_summary.csv')
    summary.to_csv(summary_path, index=False)
    print("[Diagnostic] Wrote:", summary_path, "rows:", len(summary))

    # --- Portfolio diagnostics (non-gating) ---
    port_base = _ensure_dir(os.path.join(run_dir, 'gauntlet', 'diagnostic_replay'))
    print("[Diagnostic] Running portfolio diagnostics...",
          f"(since_knowledge={args.since_knowledge}, max_concurrent={args.max_concurrent})")
    res = simulate_portfolio(
        df=df,
        out_base=port_base,
        starting_capital=args.starting_capital,
        position_size=args.position_size,
        max_concurrent=args.max_concurrent,
        since_knowledge=args.since_knowledge
    )
    print("[Diagnostic] Portfolio results:", res)

if __name__ == "__main__":
    main()
