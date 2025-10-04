from __future__ import annotations
import os
import warnings
import json
import math
from datetime import datetime
from pathlib import Path

# Suppress warnings but keep errors visible
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy.lib._function_base_impl')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy._core._methods')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy.lib._nanfunctions_impl')
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
warnings.filterwarnings('ignore', message='.*Persisting input arguments.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom <= 0 for slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in scalar divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='alpha_discovery.search.ga_core')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='alpha_discovery.eval.info_metrics.robustness')
warnings.filterwarnings('ignore', message='Field name "validate" in "Settings" shadows an attribute', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module=r".*alpha_.*\.data\.events")

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

# Set numpy error handling
np.seterr(all='ignore')

from alpha_discovery.config import settings
from alpha_discovery.features.registry import build_feature_matrix
from alpha_discovery.signals.compiler import compile_signals, check_signals_cache
from alpha_discovery.splits.npwf import make_inner_folds
from alpha_discovery.search.island_model import IslandManager
from alpha_discovery.search.fold_plan import GADataSpec, InnerFoldPlan
from alpha_discovery.engine import backtester
from alpha_discovery.eval import selection, metrics
from alpha_discovery.search.ga_core import (
    _infer_direction_from_metadata,
    _dna,
    _objective_keys,
    _compute_soft_penalties,
    _resolve_objective_value,
)
from alpha_discovery.dts.selector import DailyTradeSelector, SelectorConfig
from alpha_discovery.utils.trade_keys import canonical_signals_fingerprint

# Import loader
try:
    from alpha_discovery.data.loader import convert_excel_to_parquet
except Exception:
    convert_excel_to_parquet = None

# Thread caps
os.environ.update(
    VECLIB_MAXIMUM_THREADS='4',
    OMP_NUM_THREADS='4',
    OPENBLAS_NUM_THREADS='4',
    MKL_NUM_THREADS='4',
    NUMEXPR_NUM_THREADS='4',
    FEATURES_PAIRWISE_MAX_TICKERS='32',
    FEATURES_PAIRWISE_JOBS='8',
    PAIRWISE_MIN_PERIODS='10',
)


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load and prepare data from parquet."""
    pq = settings.data.parquet_file_path
    excel = settings.data.excel_file_path
    
    if not os.path.exists(excel):
        print(f"ERROR: Excel file not found: {excel}")
        return pd.DataFrame()
    
    # Refresh parquet from Excel
    if convert_excel_to_parquet is not None:
        print(f"Loading data from '{pq}' (refreshed from Excel)...")
        convert_excel_to_parquet()
    
    if not os.path.exists(pq):
        print(f"ERROR: Parquet not found: {pq}")
        return pd.DataFrame()
    
    df = pd.read_parquet(pq)
    
    # Normalize index
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.set_index('DATE')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Filter date range
    df = df.loc[(df.index.date >= settings.data.start_date) & (df.index.date <= settings.data.end_date)]
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    return df


def evaluate_on_window(
    individual: Tuple[str, List[str]],
    direction: str,
    signals_df: pd.DataFrame,
    master_df: pd.DataFrame,
    test_idx: pd.Index,
) -> Optional[Dict[str, Any]]:
    """Evaluate an individual on an explicit window (outer test)."""
    if individual is None or len(individual) < 2 or test_idx is None or len(test_idx) == 0:
        return None

    ticker, setup = individual[0], individual[1]
    window_signals = signals_df.loc[test_idx]

    ledger = backtester.run_setup_backtest_options(
        setup_signals=setup,
        signals_df=window_signals,
        master_df=master_df,
        direction=direction,
        exit_policy=None,
        tickers_to_run=[ticker],
    )

    if ledger is None or ledger.empty:
        return None

    ledger = ledger[ledger['trigger_date'].isin(test_idx)].copy()
    if ledger.empty:
        return None

    daily_returns = selection.portfolio_daily_returns(ledger)
    daily_returns = pd.to_numeric(daily_returns, errors='coerce').dropna()
    if daily_returns.empty:
        return None

    perf = metrics.compute_portfolio_metrics_bundle(
        daily_returns=daily_returns,
        trade_ledger=ledger,
        do_winsorize=True,
        bootstrap_B=1000,
        bootstrap_method="stationary",
        seed=int(getattr(settings.ga, 'seed', 194)),
        n_trials_for_dsr=100,
    )
    perf.update({
        "support": float(perf.get("support", 0.0)),
        "n_trades": int(len(ledger)),
    })

    return {
        "metrics": perf,
        "ledger": ledger,
        "daily_returns": daily_returns,
        "window_start": str(test_idx[0]),
        "window_end": str(test_idx[-1]),
    }


def _align_to_trading_index(index: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp:
    if target in index:
        return target
    pos = index.searchsorted(target, side='left')
    if pos >= len(index):
        pos = len(index) - 1
    elif index[pos] > target and pos > 0:
        pos -= 1
    return index[pos]


def build_trigger_map(
    priors_records: List[Dict[str, Any]],
    signals_df: pd.DataFrame,
    as_of: pd.Timestamp,
    entry_window_days: int,
    mode: str,
) -> Tuple[Dict[str, Dict[str, Any]], pd.Timestamp]:
    if signals_df.empty:
        return {}, as_of

    trading_index = signals_df.index.sort_values().unique()
    aligned_as_of = _align_to_trading_index(trading_index, pd.Timestamp(as_of))
    as_of_position = trading_index.searchsorted(aligned_as_of)
    start_pos = max(0, as_of_position - entry_window_days + 1)
    window_index = trading_index[start_pos: as_of_position + 1]
    window_slice = signals_df.loc[window_index]

    trigger_map: Dict[str, Dict[str, Any]] = {}

    for record in priors_records:
        setup_id = record.get('setup_id')
        signals = record.get('signals_list') or []
        trigger_details: Dict[str, Any] = {
            'fired_any': False,
            'fired_all': False,
            'mode': mode,
            'entry_window_days': entry_window_days,
            'signal_hits': {},
            'reason': 'no_trigger_window',
        }

        if not signals:
            trigger_details['reason'] = 'no_signals_defined'
            trigger_map[setup_id] = trigger_details
            continue

        missing = [sig for sig in signals if sig not in window_slice.columns]
        if missing:
            trigger_details['reason'] = f"missing_signals:{','.join(missing)}"
            trigger_map[setup_id] = trigger_details
            continue

        candidate_slice = window_slice[signals].astype(bool)
        any_mask = candidate_slice.any(axis=1)
        all_mask = candidate_slice.all(axis=1)

        trigger_details['signal_hits'] = {
            sig: bool(candidate_slice[sig].any())
            for sig in signals
        }

        fired_all_dates = candidate_slice.index[all_mask.values]
        fired_any_dates = candidate_slice.index[any_mask.values]
        signals_fp = canonical_signals_fingerprint(
            signals,
            trigger_details['signal_hits'],
            mode=mode,
            window_days=entry_window_days,
        )
        trigger_details['fired_all'] = len(fired_all_dates) > 0
        trigger_details['fired_any'] = len(fired_any_dates) > 0
        trigger_details['trigger_dates_all'] = [d.isoformat() for d in fired_all_dates]
        trigger_details['trigger_dates_any'] = [d.isoformat() for d in fired_any_dates]
        trigger_details['most_recent_any'] = fired_any_dates[-1].isoformat() if len(fired_any_dates) else None
        trigger_details['most_recent_all'] = fired_all_dates[-1].isoformat() if len(fired_all_dates) else None
        trigger_details['trigger_strength_all'] = float(all_mask.sum()) / max(1, len(window_index))
        trigger_details['trigger_strength_soft'] = float(sum(trigger_details['signal_hits'].values())) / max(1, len(signals))
        trigger_details['total_trigger_count'] = int(any_mask.sum())
        trigger_details['signals_fingerprint'] = signals_fp

        if trigger_details['fired_all']:
            trigger_details['reason'] = 'triggered_all'
        elif trigger_details['fired_any']:
            trigger_details['reason'] = 'triggered_partial'

        trigger_map[setup_id] = trigger_details

    return trigger_map, aligned_as_of


def create_run_dir() -> str:
    """Create timestamped run directory."""
    rd_base = settings.reporting.runs_dir
    _ensure_dir(rd_base)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(rd_base, f"pawf_npwf_seed{settings.ga.seed}_{ts}")
    _ensure_dir(run_dir)
    return run_dir


def print_header():
    """Print stylized header."""
    print("=" * 80)
    print("         ALPHA DISCOVERY - PAWF + NPWF + Options Backtesting")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration Seed: {settings.ga.seed}")
    print("-" * 80)


def main():
    """Main entry point."""
    print_header()
    main_discover_workflow()


def main_discover_workflow():
    """
    Full PAWF + NPWF workflow with options backtesting.
    
    Steps:
    1. Load data and build signals (with caching)
    2. Build PAWF outer splits (4 folds with purging)
    3. For each outer fold:
       - Build NPWF inner folds from train data
       - Run Island GA with NPWF-based backtesting fitness
       - Collect Pareto-optimal setups
    4. Aggregate and save all results
    """
    
    # Load data
    print("\n--- Loading Data ---")
    master_df = load_data()
    if master_df.empty:
        print("No data; exiting.")
        return
    
    # Build/load signals (with caching)
    print("\n--- Building Features & Signals ---")
    signals_df, signals_meta = check_signals_cache(master_df)
    
    if signals_df is None:
        print("  Cache miss - building features...")
        feature_matrix = build_feature_matrix(master_df)
        signals_df, signals_meta = compile_signals(feature_matrix)
    else:
        print(f"  ✅ Loaded {len(signals_df.columns)} cached signals")
    
    # Create run directory
    run_dir = create_run_dir()
    print(f"\n--- Run Directory: {run_dir} ---")
    
    # Build PAWF outer splits
    print("\n--- Building PAWF Outer Splits ---")
    from alpha_discovery.splits.pawf import build_pawf_splits
    from alpha_discovery.splits.ids import generate_split_id
    from alpha_discovery.adapters.features import FeatureAdapter, calculate_max_lookback_from_list
    
    adapter = FeatureAdapter()
    feature_names = adapter.list_features()
    lookback_tail = calculate_max_lookback_from_list(feature_names)
    label_horizon_days = int(max(1, getattr(settings.forecast, 'default_horizon', 5)))
    outer_test_days = int(getattr(settings.splits, 'test_window_days', 252))
    pawf_step_months = int(getattr(settings.splits, 'pawf_step_months', 3))
    embargo_cap_days = int(getattr(settings.splits, 'embargo_cap_days', 21))

    pawf_splits = build_pawf_splits(
        df=master_df,
        label_horizon_days=label_horizon_days,
        feature_lookback_tail=lookback_tail,
        min_train_months=36,        # 3 years minimum training
        test_window_days=outer_test_days,
        step_months=pawf_step_months,
        embargo_cap_days=embargo_cap_days,
        regime_version="R1"
    )

    embargo_cap = int(min(lookback_tail, 63)) if lookback_tail is not None else 0
    print(f"Created {len(pawf_splits)} PAWF outer folds")
    print(f"  Label horizon / purge: {label_horizon_days} days")
    print(f"  Feature lookback: {lookback_tail} days (embargo_cap={embargo_cap})")
    print(f"  Test window: {outer_test_days} trading days, step={pawf_step_months} months")
    
    # Get tradable tickers
    tradable_tickers = settings.data.effective_tradable_tickers
    print(f"\n--- Tradable Tickers: {len(tradable_tickers)} ---")
    print(f"  {', '.join(tradable_tickers[:10])}{'...' if len(tradable_tickers) > 10 else ''}")
    
    ga_cfg = getattr(settings, 'ga', object())
    freeze_top_n = max(1, int(getattr(ga_cfg, 'freeze_top_n', 10)))

    as_of = signals_df.index.max()
    rsd_recent_days = max(1, int(getattr(ga_cfg, 'rsd_recent_days', 10)))
    rsd_min_buffer = int(getattr(ga_cfg, 'rsd_min_buffer_days', 63))
    buffer_days = max(int(label_horizon_days + (lookback_tail or 0)), rsd_min_buffer)
    cutoff_candidate = as_of - pd.Timedelta(days=buffer_days)
    cutoff_date = _align_to_trading_index(signals_df.index, cutoff_candidate)

    recent_index = signals_df.index[-rsd_recent_days:]
    recent_slice = signals_df.loc[recent_index]
    whitelist = [
        col for col in signals_df.columns
        if recent_slice.get(col) is not None and recent_slice[col].astype(bool).any()
    ]
    seed_ratio = float(np.clip(getattr(ga_cfg, 'rsd_seed_ratio', 0.75), 0.0, 1.0))
    mutation_whitelist_prob = float(np.clip(getattr(ga_cfg, 'rsd_mutation_whitelist_prob', 0.8), 0.0, 1.0))

    rsd_stats = {
        'init_total': 0,
        'init_seeded': 0,
        'mutation_total': 0,
        'mutation_whitelist': 0,
    }
    whitelist_pct = (len(whitelist) / max(1, len(signals_df.columns))) * 100.0
    rsd_context = {
        'whitelist': set(whitelist),
        'whitelist_count': len(whitelist),
        'seed_ratio': seed_ratio if whitelist else 0.0,
        'requested_seed_ratio': seed_ratio,
        'mutation_prob': mutation_whitelist_prob,
        'stats': rsd_stats,
        'cutoff_date': cutoff_date,
        'buffer_days': buffer_days,
        'recent_days': rsd_recent_days,
        'as_of': as_of,
        'whitelist_pct': whitelist_pct,
    }
    print(
        f"[RSD] Enabled -> whitelist={len(whitelist)} ({whitelist_pct:.1f}%), "
        f"seed_ratio≈{seed_ratio*100:.1f}%, mutation_bias={mutation_whitelist_prob:.2f}"
    )
    print(
        f"[RSD] as_of={as_of.date()} buffer_days={buffer_days} cutoff={cutoff_date.date()} "
        f"recent_window={rsd_recent_days} trading days"
    )
    print(
        f"[RSD] whitelist size={len(whitelist)} ({whitelist_pct:.1f}% of signals) | "
        f"seed_ratio_target≈{seed_ratio*100:.1f}% | mutation_within_whitelist={mutation_whitelist_prob:.2f}"
    )
    if not whitelist:
        print("[RSD] ⚠️  No recent signal activity detected; falling back to global random initialization")

    use_inner_npwf = bool(getattr(ga_cfg, 'use_inner_npwf', False))
    if use_inner_npwf:
        print("\n[Config] use_inner_npwf=True → building inner NPWF folds per PAWF split")
    else:
        print("\n[Config] use_inner_npwf=False → discovery uses anchored outer train only")

    # Run GA discovery for each outer fold
    print(f"\n{'='*80}")
    banner_suffix = " + NPWF" if use_inner_npwf else ""
    print(f"RUNNING PAWF DISCOVERY{banner_suffix}")
    print(f"{'='*80}")

    if getattr(settings.ga, 'tail_fold_only', False) and pawf_splits:
        pawf_splits = pawf_splits[-1:]
        print("[Config] tail_fold_only=True → running discovery on final PAWF fold only")

    all_fold_results: List[Dict[str, Any]] = []
    split_artifacts: List[Dict[str, Any]] = []
    plan_summaries: List[Dict[str, Any]] = []
    aggregated_candidates: Dict[Tuple[str, Tuple[str, ...], int], Dict[str, Any]] = {}
    objective_keys = getattr(settings.ga, 'objectives', [
        "dsr", "bootstrap_calmar_lb", "bootstrap_profit_factor_lb"
    ])
    frozen_library: Dict[Tuple[str, Tuple[str, ...], int], Dict[str, Any]] = {}

    def record_outer_evaluation(
        dna: Tuple[str, Tuple[str, ...], int],
        individual: Tuple[str, List[str]],
        direction: str,
        split_id: str,
        fold_idx: int,
        outer_eval: Optional[Dict[str, Any]],
    ) -> None:
        if not outer_eval:
            return
        agg = aggregated_candidates.setdefault(
            dna,
            {
                "individual": individual,
                "direction": direction,
                "outer_ledgers": [],
                "outer_returns": [],
                "fold_ids": [],
                "per_fold_metrics": [],
            },
        )
        agg['individual'] = individual
        agg['direction'] = direction
        agg['outer_ledgers'].append(outer_eval['ledger'])
        agg['outer_returns'].append(outer_eval['daily_returns'])
        agg['fold_ids'].append(split_id)
        fold_metric = dict(outer_eval.get('metrics', {}))
        fold_metric['split_id'] = split_id
        fold_metric['fold_idx'] = fold_idx
        agg['per_fold_metrics'].append(fold_metric)

    def compute_objective_score(metrics: Optional[Dict[str, Any]]) -> float:
        if not metrics:
            return -np.inf
        total = 0.0
        for key in objective_keys:
            val, _ = _resolve_objective_value(metrics, key)
            total += val
        return float(total)

    for fold_idx, split_spec in enumerate(pawf_splits, 1):
        split_id = generate_split_id(split_spec)

        outer_train_idx = master_df.loc[split_spec.train_start:split_spec.train_end].index.unique().sort_values()
        outer_test_idx = master_df.loc[split_spec.test_start:split_spec.test_end].index.unique().sort_values()

        if len(outer_train_idx) == 0 or len(outer_test_idx) == 0:
            print(f"\n[Fold {fold_idx}] Skipping split {split_id} due to empty train/test window.")
            continue

        effective_train_idx = outer_train_idx[outer_train_idx <= cutoff_date]
        if len(effective_train_idx) == 0:
            print(
                f"\n[Fold {fold_idx}] Skipping split {split_id} — no training data remains before cutoff {cutoff_date.date()}"
            )
            continue

        trimmed = len(effective_train_idx) < len(outer_train_idx)
        outer_train_idx = effective_train_idx

        print(f"\n{'='*80}")
        print(f"OUTER FOLD {fold_idx}/{len(pawf_splits)}: {split_id}")
        print(f"{'='*80}")
        print(f"Training Period: {outer_train_idx[0]} to {outer_train_idx[-1]} ({len(outer_train_idx)} days)")
        if trimmed:
            print(
                f"  [RSD] Train end trimmed to {outer_train_idx[-1]} to respect cutoff {cutoff_date.date()}"
            )
        print(f"Testing  Period: {outer_test_idx[0]} to {outer_test_idx[-1]} ({len(outer_test_idx)} days)")
        print(
            f"  [Fold {fold_idx}] P={split_spec.purge_days} EMB={split_spec.embargo_days} "
            f"TestDays={len(outer_test_idx)}"
        )

        plan: Optional[GADataSpec] = None
        inner_fold_plans: List[InnerFoldPlan] = []

        if use_inner_npwf:
            print(f"\n[Fold {fold_idx}] Building NPWF inner folds...")
            df_train_outer = master_df.loc[outer_train_idx]
            inner_folds_raw = make_inner_folds(
                df_train_outer=df_train_outer,
                label_horizon_days=label_horizon_days,
                feature_lookback_tail=lookback_tail,
                k_folds=3,
            )

            for inner_idx, (fold_train_idx, fold_test_idx) in enumerate(inner_folds_raw):
                fold_train_idx = pd.Index(fold_train_idx).intersection(outer_train_idx)
                fold_test_idx = pd.Index(fold_test_idx).intersection(outer_train_idx)
                if len(fold_test_idx) == 0:
                    continue
                inner_fold_plans.append(
                    InnerFoldPlan(
                        fold_id=f"{split_id}_inner_{inner_idx:02d}",
                        train_idx=fold_train_idx,
                        test_idx=fold_test_idx,
                    )
                )

            if not inner_fold_plans:
                print(f"  ⚠️  No valid inner folds for {split_id}; falling back to outer-only fitness.")
            else:
                plan = GADataSpec(
                    outer_id=split_id,
                    train_idx=outer_train_idx,
                    test_idx=outer_test_idx,
                    inner_folds=inner_fold_plans,
                    label_horizon=label_horizon_days,
                    embargo_days=split_spec.embargo_days,
                    metadata={
                        "purge_days": split_spec.purge_days,
                        "feature_tail": lookback_tail,
                    },
                )
        else:
            print(f"\n[Fold {fold_idx}] NPWF disabled → using anchored train window only")

        plan_summary = (
            plan.summary()
            if plan is not None
            else {
                "outer_id": split_id,
                "train_span": f"{outer_train_idx[0].isoformat()}:{outer_train_idx[-1].isoformat()}:{len(outer_train_idx)}",
                "test_span": f"{outer_test_idx[0].isoformat()}:{outer_test_idx[-1].isoformat()}:{len(outer_test_idx)}",
                "n_inner_folds": len(inner_fold_plans),
                "fold_hash": None,
            }
        )
        plan_summaries.append(plan_summary)

        split_artifacts.append(
            {
                "split_id": split_id,
                "train_start": str(outer_train_idx[0]),
                "train_end": str(outer_train_idx[-1]),
                "test_start": str(outer_test_idx[0]),
                "test_end": str(outer_test_idx[-1]),
                "fold_hash": getattr(plan, 'fold_hash', None),
                "n_inner_folds": len(inner_fold_plans),
                "rsd_trimmed": bool(trimmed),
                "cutoff_date": cutoff_date.isoformat(),
                "inner_folds": [
                    {
                        "fold_id": fold.fold_id,
                        "train_start": str(fold.train_idx[0]) if len(fold.train_idx) else None,
                        "train_end": str(fold.train_idx[-1]) if len(fold.train_idx) else None,
                        "test_start": str(fold.test_idx[0]) if len(fold.test_idx) else None,
                        "test_end": str(fold.test_idx[-1]) if len(fold.test_idx) else None,
                    }
                    for fold in inner_fold_plans
                ],
            }
        )

        train_signals_df = signals_df.loc[outer_train_idx]
        train_signals_df = train_signals_df.loc[:cutoff_date]

        fitness_desc = "NPWF-based backtesting" if plan is not None else "anchored outer-train backtesting"
        print(f"\n[Fold {fold_idx}] Running Island Model GA ({fitness_desc})...")
        print(f"  Population: {settings.ga.population_size}, Generations: {settings.ga.generations}")
        print(f"  Islands: {settings.ga.n_islands}, Migration Interval: {settings.ga.migration_interval}")

        island_manager = IslandManager(
            n_islands=settings.ga.n_islands,
            n_individuals=settings.ga.population_size,
            n_generations=settings.ga.generations,
            signals_df=train_signals_df,
            signals_metadata=signals_meta,
            master_df=master_df,
            plan=plan,
            migration_interval=settings.ga.migration_interval,
            seed=settings.ga.seed + fold_idx,
            rsd_context=rsd_context,
        )

        pareto_front = island_manager.evolve()

        current_outer_evals: List[Dict[str, Any]] = []

        for setup in pareto_front:
            individual = setup.get('individual')
            if not individual:
                continue

            direction = _infer_direction_from_metadata(individual[1], signals_meta)
            outer_eval = evaluate_on_window(individual, direction, signals_df, master_df, outer_test_idx)

            dna = _dna(individual)
            record_outer_evaluation(dna, individual, direction, split_id, fold_idx, outer_eval)

            current_outer_evals.append(
                {
                    'dna': dna,
                    'individual': individual,
                    'direction': direction,
                    'outer_eval': outer_eval,
                    'setup': setup,
                }
            )

            setup['outer_split_id'] = split_id
            setup['outer_fold'] = fold_idx
            setup['train_start'] = str(outer_train_idx[0])
            setup['train_end'] = str(outer_train_idx[-1])
            setup['test_start'] = str(outer_test_idx[0])
            setup['test_end'] = str(outer_test_idx[-1])
            setup['fold_plan_hash'] = getattr(plan, 'fold_hash', None)
            setup['outer_evaluation'] = outer_eval

        # Freeze top-N candidates from this fold (based on outer evaluation)
        freeze_candidates = []
        for entry in current_outer_evals:
            outer_metrics = entry['outer_eval'].get('metrics') if entry['outer_eval'] else None
            score = compute_objective_score(outer_metrics) if outer_metrics else -np.inf
            if not entry['outer_eval']:
                continue
            freeze_candidates.append({
                'dna': entry['dna'],
                'individual': entry['individual'],
                'direction': entry['direction'],
                'outer_eval': entry['outer_eval'],
                'score': score,
                'split_id': split_id,
                'fold_idx': fold_idx,
            })

        freeze_candidates.sort(key=lambda x: x['score'], reverse=True)
        top_freeze = freeze_candidates[:freeze_top_n]
        new_freezes = 0

        for candidate in top_freeze:
            dna = candidate['dna']
            existing = frozen_library.get(dna)
            if existing is None:
                new_freezes += 1
                frozen_library[dna] = {
                    'individual': candidate['individual'],
                    'direction': candidate['direction'],
                    'origin_split_id': candidate['split_id'],
                    'origin_fold_idx': candidate['fold_idx'],
                    'frozen_score': candidate['score'],
                    'evaluations': [],
                    'evaluated_folds': set(),
                }
                existing = frozen_library[dna]
            else:
                existing['individual'] = candidate['individual']
                existing['direction'] = candidate['direction']
                existing['frozen_score'] = max(existing.get('frozen_score', candidate['score']), candidate['score'])
            if candidate['outer_eval'] and candidate['split_id'] not in existing['evaluated_folds']:
                existing['evaluations'].append({
                    'split_id': candidate['split_id'],
                    'fold_idx': candidate['fold_idx'],
                    'metrics': candidate['outer_eval']['metrics'],
                    'window_start': candidate['outer_eval'].get('window_start'),
                    'window_end': candidate['outer_eval'].get('window_end'),
                })
                existing['evaluated_folds'].add(candidate['split_id'])

        # Evaluate all frozen DNAs on this fold's test window (if not already evaluated)
        for dna, frozen_entry in frozen_library.items():
            evaluated_folds = frozen_entry.setdefault('evaluated_folds', set())
            if split_id in evaluated_folds:
                continue
            eval_result = evaluate_on_window(
                frozen_entry['individual'],
                frozen_entry['direction'],
                signals_df,
                master_df,
                outer_test_idx,
            )
            record_outer_evaluation(dna, frozen_entry['individual'], frozen_entry['direction'], split_id, fold_idx, eval_result)
            frozen_entry.setdefault('evaluations', []).append({
                'split_id': split_id,
                'fold_idx': fold_idx,
                'metrics': eval_result.get('metrics') if eval_result else None,
                'window_start': eval_result.get('window_start') if eval_result else str(outer_test_idx[0]),
                'window_end': eval_result.get('window_end') if eval_result else str(outer_test_idx[-1]),
            })
            evaluated_folds.add(split_id)

        print(
            f"  [Fold {fold_idx}] Frozen {len(top_freeze)} setups this fold (new={new_freezes}), "
            f"library size={len(frozen_library)}"
        )

        all_fold_results.extend(pareto_front)

        print(f"\n[Fold {fold_idx}] Discovered {len(pareto_front)} Pareto-optimal setups")

    rsd_stats = rsd_context.get('stats', {})
    seeded_pct = 0.0
    if rsd_stats.get('init_total'):
        seeded_pct = (rsd_stats['init_seeded'] / max(1, rsd_stats['init_total'])) * 100.0
    mutation_pct = 0.0
    if rsd_stats.get('mutation_total'):
        mutation_pct = (rsd_stats['mutation_whitelist'] / max(1, rsd_stats['mutation_total'])) * 100.0
    rsd_context['seeded_pct'] = seeded_pct
    rsd_context['mutation_pct'] = mutation_pct
    print(
        f"[RSD] Initialization seeded {rsd_stats.get('init_seeded', 0)}/"
        f"{rsd_stats.get('init_total', 0)} ({seeded_pct:.1f}%) from whitelist"
    )
    if rsd_stats.get('mutation_total', 0):
        print(
            f"[RSD] Mutation whitelist usage: {rsd_stats.get('mutation_whitelist', 0)}/"
            f"{rsd_stats.get('mutation_total', 0)} ({mutation_pct:.1f}%)"
        )
    else:
        print("[RSD] Mutation whitelist usage: no signal mutations recorded")

    aggregated_results: List[Dict[str, Any]] = []
    total_outer_splits = max(1, len(split_artifacts))
    for dna, data in aggregated_candidates.items():
        if not data['outer_returns']:
            continue

        combined_returns = pd.concat(data['outer_returns'], axis=0)
        combined_returns = pd.to_numeric(combined_returns, errors='coerce').dropna()
        if combined_returns.empty:
            continue
        combined_returns = combined_returns.groupby(combined_returns.index).mean().sort_index()

        combined_ledger = pd.concat(data['outer_ledgers'], ignore_index=True)

        sanitized_returns = combined_returns.clip(lower=-0.95, upper=10.0)
        sanitized_ledger = combined_ledger.copy()
        if 'pnl_pct' in sanitized_ledger.columns:
            sanitized_ledger['pnl_pct'] = pd.to_numeric(sanitized_ledger['pnl_pct'], errors='coerce').clip(lower=-0.95, upper=10.0)

        agg_perf = metrics.compute_portfolio_metrics_bundle(
            daily_returns=sanitized_returns,
            trade_ledger=sanitized_ledger,
            do_winsorize=True,
            bootstrap_B=1000,
            bootstrap_method="stationary",
            seed=int(getattr(settings.ga, 'seed', 194)),
            n_trials_for_dsr=100,
        )
        agg_perf.update({
            "support": float(agg_perf.get("support", 0.0)),
            "n_trades": int(len(combined_ledger)),
            "outer_split_ids": data["fold_ids"],
            "per_fold_metrics": data["per_fold_metrics"],
            "fold_count": len(data["fold_ids"]),
        })
        if not combined_ledger.empty:
            agg_perf["first_trigger"] = str(combined_ledger['trigger_date'].min())
            agg_perf["last_trigger"] = str(combined_ledger['trigger_date'].max())
        coverage_ratio_target = float(getattr(settings.ga, 'min_supported_fold_ratio', 0.1))
        min_outer_absolute = int(getattr(settings.ga, 'min_outer_folds', 2))
        required_outer = max(min_outer_absolute, math.ceil(total_outer_splits * coverage_ratio_target))

        coverage_ratio = agg_perf.get("fold_coverage_ratio")
        if coverage_ratio is None:
            coverage_ratio = float(agg_perf["fold_count"]) / float(total_outer_splits)
        coverage_shortfall = max(0, required_outer - agg_perf["fold_count"])
        agg_perf["fold_coverage_ratio"] = float(coverage_ratio)
        agg_perf["coverage_target_ratio"] = coverage_ratio_target
        agg_perf["coverage_min_folds"] = required_outer
        agg_perf["coverage_shortfall"] = coverage_shortfall

        total_trades = int(agg_perf.get("n_trades", 0) or 0)
        if agg_perf.get("support", 0) <= 0 or total_trades <= 0:
            continue

        penalty_scalar, penalty_details, reasons = _compute_soft_penalties(
            agg_perf,
            settings.ga,
            coverage_ratio=coverage_ratio,
            coverage_target=coverage_ratio_target,
            min_total_trades_required=int(getattr(settings.ga, 'min_total_trades', 20)),
        )
        agg_perf["soft_penalties"] = penalty_details
        agg_perf["penalty_scalar"] = penalty_scalar
        agg_perf["eligibility_reasons"] = list(dict.fromkeys(reasons))
        agg_perf["eligible"] = True
        agg_perf["fatal_reasons"] = []

        objective_sources = {}
        objectives = []
        for key in objective_keys:
            original_val = agg_perf.get(key)
            agg_perf[f"{key}_raw"] = original_val
            sanitized_val, source = _resolve_objective_value(agg_perf, key)
            agg_perf[key] = sanitized_val
            agg_perf[f"{key}_score"] = sanitized_val
            objective_sources[key] = source
            if sanitized_val <= -998.0:
                objectives.append(sanitized_val)
            else:
                objectives.append(float(sanitized_val / penalty_scalar))
        agg_perf["objective_sources"] = objective_sources

        signals_list = data["individual"][1] if len(data["individual"]) > 1 else []
        signals_fp = canonical_signals_fingerprint(signals_list)
        agg_perf["signals_list"] = list(signals_list)
        agg_perf["signals_fingerprint"] = signals_fp

        aggregated_results.append({
            "individual": data["individual"],
            "direction": data["direction"],
            "metrics": agg_perf,
            "objectives": objectives,
            "trade_ledger": combined_ledger,
            "daily_returns": sanitized_returns,
        })

    # --- Enforce unique signal fingerprints across tickers ---
    signal_best: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    suppressed_signal_duplicates: List[Dict[str, Any]] = []

    def _signal_score(entry: Dict[str, Any]) -> float:
        m = entry.get('metrics', {})
        for key in ("dsr_score", "dsr", "bootstrap_calmar_lb_score", "bootstrap_calmar_lb"):
            val = m.get(key)
            if val is None:
                continue
            try:
                val_f = float(val)
            except (TypeError, ValueError):
                continue
            if np.isfinite(val_f):
                return val_f
        return float('-inf')

    for entry in aggregated_results:
        individual = entry.get('individual', ())
        ticker = individual[0] if individual else None
        signals_list = individual[1] if len(individual) > 1 else []
        horizon = individual[2] if len(individual) > 2 else -1
        direction = entry.get('direction') or 'long'
        metrics_payload = entry.get('metrics', {})
        metrics_payload.setdefault('signals_list', list(signals_list))
        metrics_payload.setdefault('signals_fingerprint', canonical_signals_fingerprint(signals_list))

        key = (metrics_payload['signals_fingerprint'], int(horizon), str(direction))
        score_val = _signal_score(entry)
        metrics_payload['signal_dup_score'] = float(score_val)
        metrics_payload['signal_dup_status'] = 'kept'
        metrics_payload['signal_dup_reason'] = None
        metrics_payload['signal_dup_replacement'] = None
        metrics_payload['signal_dup_ticker'] = ticker

        existing = signal_best.get(key)
        if existing is None:
            signal_best[key] = {'entry': entry, 'score': score_val}
            continue

        if score_val > existing['score']:
            prev_entry = existing['entry']
            prev_metrics = prev_entry.get('metrics', {})
            prev_metrics['signal_dup_status'] = 'suppressed'
            prev_metrics['signal_dup_reason'] = 'signal_conflict_replaced'
            prev_metrics['signal_dup_replacement'] = ticker
            suppressed_signal_duplicates.append(prev_entry)

            signal_best[key] = {'entry': entry, 'score': score_val}
        else:
            metrics_payload['signal_dup_status'] = 'suppressed'
            metrics_payload['signal_dup_reason'] = 'signal_conflict_weaker'
            metrics_payload['signal_dup_replacement'] = existing['entry'].get('individual', [None])[0]
            suppressed_signal_duplicates.append(entry)

    aggregated_results = [payload['entry'] for payload in signal_best.values()]
    total_signal_dups = len(suppressed_signal_duplicates)

    print(f"\n{'='*80}")
    print(f"PAWF + NPWF DISCOVERY COMPLETE")
    print(f"{'='*80}")
    print(f"Total fold-level setups discovered: {len(all_fold_results)}")
    print(f"Aggregated unique setups: {len(aggregated_results)}")
    if total_signal_dups:
        print(f"Signal-level duplicates suppressed: {total_signal_dups}")

    # Save results
    print("\n--- Saving Results ---")
    from alpha_discovery.reporting.artifacts import save_results
    
    results_df = pd.DataFrame(aggregated_results)

    # Build splits object for save_results (simple dict representation)
    fold_results_serializable: List[Dict[str, Any]] = []
    for entry in all_fold_results:
        summary = {
            "individual": str(entry.get("individual")),
            "outer_split_id": entry.get("outer_split_id"),
            "outer_fold": entry.get("outer_fold"),
            "objectives": entry.get("objectives"),
            "metrics": entry.get("metrics"),
        }
        outer_eval = entry.get("outer_evaluation")
        if outer_eval:
            summary["outer_metrics"] = outer_eval.get("metrics")
            summary["outer_window"] = {
                "start": outer_eval.get("window_start"),
                "end": outer_eval.get("window_end"),
            }
            ledger = outer_eval.get("ledger")
            summary["outer_n_trades"] = int(len(ledger)) if ledger is not None else 0
        fold_results_serializable.append(summary)

    split_type = 'PAWF+NPWF' if use_inner_npwf else 'PAWF'
    cutoff_respected = True
    as_of_ts = rsd_context.get('as_of')
    rsd_summary = {
        'as_of': as_of_ts.isoformat() if isinstance(as_of_ts, pd.Timestamp) else None,
        'buffer_days': rsd_context.get('buffer_days'),
        'cutoff_date': rsd_context.get('cutoff_date').isoformat() if isinstance(rsd_context.get('cutoff_date'), pd.Timestamp) else None,
        'recent_days': rsd_context.get('recent_days'),
        'whitelist_size': rsd_context.get('whitelist_count', 0),
        'whitelist_pct': rsd_context.get('whitelist_pct'),
        'seed_ratio_requested': rsd_context.get('requested_seed_ratio'),
        'seed_ratio_effective_pct': rsd_context.get('seeded_pct'),
        'mutation_whitelist_pct': rsd_context.get('mutation_pct'),
    }

    splits_dict = {
        'outer_split_count': len(split_artifacts),
        'type': split_type,
        'tail_aligned': bool(tail_aligned),
        'label_horizon_days': label_horizon_days,
        'feature_lookback_tail': lookback_tail,
        'aggregated_candidate_count': len(aggregated_results),
        'fold_level_count': len(all_fold_results),
        'splits': split_artifacts,
        'fold_level_summaries': fold_results_serializable,
        'cutoff_respected': cutoff_respected,
        'rsd_summary': rsd_summary,
        'frozen_library_stats': frozen_library_stats,
        'frozen_library': frozen_library_serializable,
        'signal_duplicate_suppressed': [
            {
                'signals_fingerprint': entry.get('metrics', {}).get('signals_fingerprint'),
                'suppressed_ticker': entry.get('individual', [None])[0],
                'replacement_ticker': entry.get('metrics', {}).get('signal_dup_replacement'),
                'score': entry.get('metrics', {}).get('signal_dup_score'),
                'reason': entry.get('metrics', {}).get('signal_dup_reason'),
            }
            for entry in suppressed_signal_duplicates
        ],
        'plans': plan_summaries,
        'aggregated_results_path': 'aggregated_results.json',
        'fold_level_results_path': 'fold_level_results.json',
        'dts_summary': summary_counts,
        'dts_hud_line': hud_line_summary,
        'dts_near_miss': near_miss,
    }
    
    artifact_info = save_results(results_df, signals_meta, run_dir, splits_dict, settings)

    sanitized_results = results_df.drop(columns=['trade_ledger', 'daily_returns'], errors='ignore')
    aggregated_json_path = os.path.join(run_dir, 'aggregated_results.json')
    serialized = []
    for row in sanitized_results.to_dict(orient='records'):
        clean_row = {}
        for key, value in row.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float) and not np.isfinite(value):
                    clean_row[key] = None
                else:
                    clean_row[key] = value
            elif isinstance(value, (str, bool)) or value is None:
                clean_row[key] = value
            else:
                clean_row[key] = json.loads(json.dumps(value, default=str))
        serialized.append(clean_row)
    with open(aggregated_json_path, 'w') as f:
        json.dump(serialized, f, indent=2)
    print(f"Aggregated results JSON saved to: {aggregated_json_path}")

    fold_json_path = os.path.join(run_dir, 'fold_level_results.json')
    with open(fold_json_path, 'w') as f:
        json.dump(fold_results_serializable, f, indent=2)
    print(f"Fold-level summaries saved to: {fold_json_path}")

    print(f"\n{'='*80}")
    print(f"✅ DISCOVERY COMPLETE!")
    print(f"{'='*80}")
    print(f"📁 Results saved to: {run_dir}")
    print(f"   - Options summary: {run_dir}/options_summary.csv")
    print(f"   - Options trade ledger: {run_dir}/options_trade_ledger.csv")
    priors_records = artifact_info.get('priors_records', [])
    dts_cfg = getattr(settings, 'dts', None)
    default_selector_cfg = SelectorConfig()
    selector_config = SelectorConfig(
        entry_window_days=getattr(dts_cfg, 'entry_window_days', default_selector_cfg.entry_window_days),
        recent_trades_floor=getattr(dts_cfg, 'recent_trades_floor', default_selector_cfg.recent_trades_floor),
        min_total_trades=getattr(dts_cfg, 'min_total_trades', default_selector_cfg.min_total_trades),
        dormancy_half_life_days=getattr(dts_cfg, 'dormancy_half_life_days', default_selector_cfg.dormancy_half_life_days),
        mode=getattr(dts_cfg, 'mode', default_selector_cfg.mode),
        soft_and_penalty=getattr(dts_cfg, 'soft_and_penalty', default_selector_cfg.soft_and_penalty),
        friction_penalty=getattr(dts_cfg, 'friction_penalty', default_selector_cfg.friction_penalty),
    )
    trigger_map, aligned_as_of = build_trigger_map(
        priors_records=priors_records,
        signals_df=signals_df,
        as_of=master_df.index.max(),
        entry_window_days=selector_config.entry_window_days,
        mode=selector_config.mode,
    )

    selector = DailyTradeSelector(
        priors=priors_records,
        trigger_map=trigger_map,
        as_of=aligned_as_of,
        config=selector_config,
    )
    dts_live_df = selector.run()
    dts_full_df = selector.last_all_candidates
    why_counts = selector.why_counts()
    summary_counts = selector.summary()
    near_miss = selector.near_miss_candidates()

    dts_live_path = os.path.join(run_dir, 'dts_today.csv')
    dts_live_json = os.path.join(run_dir, 'dts_today.json')
    dts_summary_path = os.path.join(run_dir, 'dts_today_diagnostics.csv')
    dts_summary_json = os.path.join(run_dir, 'dts_today_diagnostics.json')
    dts_why_path = os.path.join(run_dir, 'dts_today_why_counts.json')
    dts_summary_counts_path = os.path.join(run_dir, 'dts_today_summary.json')
    dts_near_miss_path = os.path.join(run_dir, 'dts_today_near_miss.json')

    dts_live_df.to_csv(dts_live_path, index=False)
    with open(dts_live_json, 'w') as f:
        json.dump(dts_live_df.to_dict(orient='records'), f, indent=2, default=str)

    dts_full_df.to_csv(dts_summary_path, index=False)
    with open(dts_summary_json, 'w') as f:
        json.dump(dts_full_df.to_dict(orient='records'), f, indent=2, default=str)

    with open(dts_why_path, 'w') as f:
        json.dump(why_counts, f, indent=2, default=str)
    with open(dts_summary_counts_path, 'w') as f:
        json.dump(summary_counts, f, indent=2, default=str)
    with open(dts_near_miss_path, 'w') as f:
        json.dump(near_miss, f, indent=2, default=str)

    print(f"   - DTS live slate: {dts_live_path}")
    print(f"   - DTS live JSON: {dts_live_json}")
    print(f"   - DTS diagnostics: {dts_summary_path}")
    print(f"   - DTS diagnostics JSON: {dts_summary_json}")
    print(f"   - DTS summary JSON: {dts_summary_counts_path}")
    print(f"   - DTS near-miss: {dts_near_miss_path}")
    if why_counts:
        print(f"   - DTS why-not counts: {why_counts}")
    if summary_counts:
        hud_line = (
            f"[DTS HUD] checked={summary_counts.get('n_checked', 0)} "
            f"recent_trigger={summary_counts.get('n_recent_trigger', 0)} "
            f"fatal={summary_counts.get('n_blocked_fatal', 0)} "
            f"soft={summary_counts.get('n_soft_flagged', 0)} "
            f"bootstrap={summary_counts.get('n_scored_with_bootstrap', 0)} "
            f"point_estimate={summary_counts.get('n_scored_point_estimate', 0)} "
            f"selected={summary_counts.get('n_final_selected', 0)}"
        )
        print(hud_line)
    if summary_counts.get('n_final_selected', 0) == 0 and near_miss:
        print("[DTS HUD] Top near-miss candidates:")
        for miss in near_miss:
            print(
                "    ",
                miss.get('setup_id'),
                miss.get('ticker'),
                f"score={miss.get('candidate_score', 0.0):.4f}",
                f"reasons={miss.get('reasons', [])}"
            )
    print(f"   - Config: {run_dir}/config.json")
    print(f"\n🎯 DTS candidates JSON/CSV are ready for the live selector.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
