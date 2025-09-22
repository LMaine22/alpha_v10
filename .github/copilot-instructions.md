Purpose
Provide short, actionable guidance so an AI coding assistant can be immediately productive in this repository.

Quick orientation
- Top-level entry: `main.py` orchestrates the full discovery -> validation -> reporting pipeline.
- Core package: `alpha_discovery/` contains the system: `data/`, `features/`, `signals/`, `search/`, `engine/`, `eval/`, `reporting/`.
- Central configuration singleton: `alpha_discovery/config.py` (use `settings` to read/modify run knobs).

Key concepts an AI must know
- Individual DNA: an individual is a tuple `(ticker, [signal_ids])`. The canonical helper is `_dna(individual)` in `alpha_discovery/search/ga_core.py`.
- Evaluation flow: GA -> `_evaluate_one_setup` -> `_calculate_objectives` -> validation pipeline in `alpha_discovery/eval/validation.py`.
- Backtesting: options-backtest logic lives under `alpha_discovery/engine/` (see `bt_core.py` and `bt_runtime.py`); pricing/IV mapping is cache-heavy and expects `master_df` with ticker-prefixed price/IV columns.

Common tasks & how to run locally
- Quick run (full pipeline): python `main.py`. The script will attempt to build parquet from Excel if missing (see `main.py:load_data`).
- To run only discovery: set `settings.run_mode = 'discover'` or pass/modify configuration before calling `main()`.
- Data files: expected paths are in `settings.data` (defaults in `alpha_discovery/config.py`): `data_store/raw/*.xlsx` and `data_store/processed/bb_data.parquet`.

Repository conventions
- Config-driven: change behaviour by editing `settings` in `alpha_discovery/config.py` rather than scattering constants.
- Date index: dataframes use a DatetimeIndex (column `DATE` normalized in `main.py`); many functions rely on business-day ranges.
- Signal columns: `signals_df` columns are signal IDs (strings) and are expected to be boolean-like; primitive signals are compiled by `alpha_discovery/signals/compiler.py`.
- Deterministic evaluations: GA seeds individual evaluations using a deterministic adler32-based seed (`_seed_for_individual` in `nsga.py`) — avoid changing randomness without updating this logic.

Performance / caching notes
- `alpha_discovery/engine/bt_runtime.py` uses in-memory caches keyed by a token derived from the input DataFrame. Use `_maybe_reset_caches(master_df)` when reusing different master frames in the same process.
- Heavy numeric code may respect BLAS env vars (the project sets VECLIB/OMP/OPENBLAS caps in `main.py`). Tests and parallel evaluation use `joblib` — use `threadpool_limits` context if adding heavy CPU work.

Patterns & gotchas (explicit examples)
- Avoid mutating `settings` mid-evaluation; GA builds `exit_policy = _exit_policy_from_settings()` once and reuses it across evaluations (`nsga.py`).
- Individual deduplication: NSGA relies on `_dna` + set membership to avoid clones. When generating or mutating individuals, ensure stable ordering of signals (use sorted lists) so `_dna` remains stable.
- Price path horizon clamping: `_get_or_build_price_path` constrains horizon to available data (see `bt_runtime.py`) — do not assume future dates exist.

Where to look for specific changes
- Add new feature: `alpha_discovery/features/core.py` + register in `features/registry.py`.
- Add new primitive signal: implement transformer in `signals/compiler.py` and add metadata consumed by `reporting/display_utils.py`.
- Add GA objective: implement metric in `alpha_discovery/search/ga_core.py` (see how `crps_neg` and other negated objectives are created) and add its string name to `settings.ga.objectives`.

Testing & debugging tips
- Unit tests live under `tests/` — run the subset quickly using pytest (-k to filter). There are fixtures for small DataFrames under `tests/fixtures/`.
- For debugging GA runs, set `settings.ga.debug_sequential=True` to avoid parallelism and make prints deterministic.
- To inspect trade ledgers, call `run_setup_backtest_options(...)` in `engine/bt_core.py` on a small subset of signals and print the returned ledger.

What to avoid
- Do not change the meaning of DNA tuples (ticker + signal list order) without updating `_dna` and deterministic seeding — it will silently break reproducibility and caching.
- Avoid relying on global mutable state (module-level caches in `bt_runtime.py`); prefer explicit `_maybe_reset_caches(master_df)` when switching data.

If you need more detail
- Ask for specific tasks (e.g., "add new GA objective X", "add a primitive signal Y", or "optimize backtester cache behavior"). Provide the target file and a short acceptance test (small DataFrame or expected output) and I'll implement.
