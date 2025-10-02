# Alpha Discovery Engine v10 ("Helios")

> A comprehensive, fail‑closed quantitative research & alpha discovery framework. It couples adaptive, leakage‑resistant cross‑validation with deterministic evolutionary search, multi‑axis robustness scoring (ELV + Hart Index), provenance logging, and performance‑aware feature generation scaling. Designed for auditability, reproducibility, and rapid extension under experimental load.

---
## 1. Core Philosophy (Deep Rationale)

The platform is engineered around a **research audit chain**: every transformation that influences model selection must be explainable, reproducible, and falsifiable. The system enforces *defensive transparency*: if support is insufficient, we do not “fill in” performance — we surface absence.

| Principle | Implementation Detail | Deep Rationale | Failure Avoided |
|-----------|----------------------|----------------|-----------------|
| Fail‑Closed Evaluation | Folds w/ insufficient triggers produce no synthetic surrogate metrics; explicit skip reason attached & logged | “Optimistic hallucination” of robustness is statistically toxic; absence of evidence must not become positive evidence | Survival bias via default penalties / smoothing |
| Determinism | Adler32 DNA → RNG seeds; ordered signal lists; stable column hashing for caches | Enables binary diff reproducibility; unblocks forensic comparison of runs | Non‑replayable fitness landscapes |
| Provenance | Early run_dir creation + `reproducibility.json` + header injection in Pareto CSV | Downstream artifacts carry lineage context (seed/objectives/hash) | Orphaned artifacts without lineage |
| Explicit Objective Normalization | Central sign inversion map; all GA operates in maximize space | Reduces cognitive & code complexity; eliminates accidental minimization drift | Divergent optimization semantics |
| Transparency | Sparsity diagnostics, fold skip reasons, Hart Index component usage report | Empirical insight into *why* a result exists (or not) | Opaque model acceptance |
| Performance with Guardrails | Memory footprint job clamping, slimmed pairwise view, on‑disk caching, thread backend | Maintains interactive iteration speed without unstable over‑parallelization | Degenerate stalls from fork/pickle explosion |
| Extensibility via Registries | Feature, macro, pairwise, objective, event interaction registries | Declarative addition; low surface area for regressions | Sprawling ad‑hoc feature spigots |
| Single Source of Truth for Config | `settings` + env gates layered (code > env) | Enforces testable, introspectable configuration space | Divergent implicit defaults |
| Structured Robustness Scoring | ELV + Hart Index derived from orthogonal metric families | Avoids over‑weighting pure performance; encodes reliability dimensions | Performance-only false positives |

**Non‑Goals** (intentionally excluded):
1. Hidden performance smoothing (e.g., implicit L2 fallback when volatility too low).
2. Silent substitution of missing metrics with population means.
3. Monolithic black‑box pipelines — each phase emits its own audit artifacts.
4. “Hero” hyperparameter hunts; instead we emphasize *structural* reliability.

---
## 2. End‑to‑End Pipeline (Annotated Dataflow)

```
┌──────────────────────┐
│ Raw Data (Excel)     │  daily instrument + event sheets
└──────────┬───────────┘
           │ convert / refresh (idempotent)
           ▼
┌──────────────────────┐
│ Parquet Canonical    │  normalized date index, typed columns
└──────────┬───────────┘
           │ load_data(): date filtering + monotonic guarantee
           ▼
┌──────────────────────┐
│ Feature Matrix (X)   │  macro + single + pairwise + events
└──────────┬───────────┘
           │ compile_signals(): boolean primitives
           ▼
┌──────────────────────┐
│ Hybrid Splits / CPCV │  discovery vs OOS, purged folds
└──────────┬───────────┘
           │ GA evolution (multi-objective maximize space)
           ▼
┌──────────────────────┐
│ Candidate Pareto     │  front + DNA provenance
└──────────┬───────────┘
           │ run_full_pipeline(): strict eval
           ▼
┌──────────────────────┐
│ OOS / Gauntlet Eval  │  metrics, skip reasons retained
└──────────┬───────────┘
           │ ELV + Hart Index computation
           ▼
┌──────────────────────┐
│ Scored Setups        │  resilience & trust layers
└──────────┬───────────┘
           │ post_simulation (ledger, correlations, causality)
           ▼
┌──────────────────────┐
│ Reporting Artifacts  │  forecast slate, tradeables, diagnostics
└──────────────────────┘
```

Each arrow boundary is **side‑effect documented** — intermediate artifacts allow partial replays (e.g., re‑computing Hart Index on archived evaluation results without re‑running GA).

---
## 3. Data Layer (Deep Dive)

### 3.1 Schema Harmonization
The loader enforces canonical naming patterns: `<TICKER>_<FIELD>`. Derived convenience accessors (`_col`) insulate feature definitions from raw naming drift. This separation allows upstream ingestion refactors without rewriting feature logic.

### 3.2 Event Normalization Philosophy
Event features are intentionally sparse; absence of an event is *meaningful*. We refuse mean‑imputation or forward‐fill for structural event markers (e.g., `EV_after_surprise_z`), as that would blur temporal locality of information shocks. Instead, we propagate them raw, only applying a leak‑safe shift where necessary.

### 3.3 Label Pair Generation
Horizon pairs (t0, tH) are cached by a *stable hash of index + horizon spec*. The caching contract:
1. Deterministic: identical index + horizons => identical cache key.
2. Integrity: key invalidation occurs if horizon list or index changes length/hash.
3. Safety: calendar clamping prevents horizon projecting beyond final available date; truncated horizons are explicitly shorter rather than padded.

### 3.4 Data Quality Considerations
| Issue | Strategy |
|-------|----------|
| Mixed frequency anomalies | Enforce daily resample; drop intraday leakage | 
| Stale rows (unchanged blocks) | Rolling functions inherently unaffected; no synthetic jitter | 
| Phantom zeros (vendor artifacts) | Prefer `to_numeric(errors='coerce')` then treat as NaN | 
| Index discontinuities | CPCV horizon expansion aware of missing days; avoids false density inflation | 

---
## 4. Feature Engineering (Explanatory Mechanics)

### 4.1 Leakage Control Model
All forward‑looking transforms (volatility, z‑scores, correlations) operate on strictly historical windows due to `.shift(1)` post‑computation. The shift occurs **after** expensive transforms to avoid redundant windowing while still guaranteeing no same‑day label contamination.

### 4.2 Macro & Cross‑Asset Constructs
Macro features attempt to encode relative regime signals (e.g., `macro.rty_over_spx`, yield curve slope dynamics) rather than absolute levels. Where spreads are used, we ensure scale stability (differences of rates vs. ratios of price indices) so z‑scoring does not produce cross‑instrument comparability illusions.

### 4.3 Pairwise Subsystem (Algorithmic Steps)
1. Derive minimal sub‑frame: returns/proxies for target & benchmark.
2. Compute fast hash → (target, benchmark, content hash) key.
3. Consult joblib cache: if hit → reuse; if miss → compute.
4. Each pair spec invoked via `call_pair_spec` injecting `min_periods` if accepted — prevents brittle mass rewrites.
5. Results shift(1) and merged into global feature dict.
6. Heartbeat prints progress every K completed pairs.

**Why Threads?** CPU‑bound correlation/rolling ops in NumPy/Pandas release the GIL at the C extension boundary (e.g., internal `ndarray` loops). Empirically this yields concurrency without serialization cost of process forking and IPC. For extremely heavy linear algebra we could introduce hybrid scheduling (future).

### 4.4 Safe Rolling Helpers
`pairwise_utils.safe_rolling_corr` wraps `.rolling(..., min_periods=...)` to eliminate silent early NaN cascades while still enforcing a minimal statistical support threshold. This reduces noisy high‑volatility first‑window artifacts.

### 4.5 Cross‑Sectional Adjustments
Sector neutrality is implemented as intra‑sector demeaning before ranking. We avoid full z‑scoring pre‑ranking because percentile ranks are distribution‑agnostic, mitigating skew distortions while still controlling structural sector tilts.

### 4.6 Quality Gate Construction
Liquidity flags rely on *co‑positive* z‑scores of volume and turnover, intentionally more conservative than single dimension gating. Options availability is a forward‑looking eligibility guard for strategies requiring volatility surface presence.

### 4.7 Complexity Metric Strategy (Future Re‑Enablement)
Moved behind a performance gate; plan: isolate complexity transforms into streaming chunk evaluation with standardized min_periods and optional incremental caching keyed by (ticker, window, cut length).

---
## 5. Signal Compilation Layer
Signals are **boolean abstractions** over numeric features; examples: threshold crossings, percentile gates, divergence conditions. Compilation rules:
1. Preserve monotonicity: no label referencing inside a signal.
2. Explicit naming to reflect predicate semantics (e.g., `mom_21_gt_0` not `sig42`).
3. Fire count collected → ultra‑sparse signals removed (below `settings.validation.min_signal_fires`) preventing combinatorial explosion during GA.

**Why Boolean?** Increases genetic operator interpretability; combination logic becomes discrete set arithmetic instead of latent continuous ensembles, improving reproducibility and interpretability of evolved DNA.

---
## 6. Evolutionary Search (Full Mechanics)

### 6.1 DNA & Seed Determinism
```
DNA = (ticker, tuple(sorted(signal_ids)))
seed = Adler32(str(DNA).encode()) % 2**32
```
This seed parameterizes any stochastic sub‑routine (mutation selection, crossover ordering) ensuring the same candidate under identical signal set reproduces bit‑for‑bit evaluation trajectories.

### 6.2 Population Dynamics
| Component | Implementation | Notes |
|-----------|---------------|-------|
| Initialization | Random subsets of eligible signals per ticker respecting min size | Biased away from empty genomes |
| Selection | NSGA (Pareto dominance sorting) | Maintains objective frontier diversity |
| Crossover | Set union / intersection operations controlling size drift | Avoids unbounded length growth |
| Mutation | Add / drop / swap signals with probability schedule | Controlled by seed for determinism |
| Replacement | Elitism applied per island then migration | Preserves top frontier viability |

### 6.3 Objective Transformation
Every raw objective `o` with orientation metadata (`maximize` or `minimize`) becomes `o' = sign * o + offset` such that all effective objectives are maximized. For ratio/penalty style metrics we *exclude* ad‑hoc normalization; all transformations are in a central registry for auditable review.

### 6.4 Constraint Handling (Fail‑Closed)
Infeasible candidate (e.g., zero triggers across all discovery folds) → evaluation returns explicit infeasible flag; candidate excluded from dominance comparison rather than artificially penalized with synthetic numeric placeholders. This avoids ordering artifacts that would otherwise “hide” structural emptiness.

---
## 7. Validation & CPCV++ (Algorithmic Detail)

### 7.1 Classical Issues Addressed
| Problem | Classical Impact | Our Mitigation |
|---------|------------------|----------------|
| Temporal leakage | Inflated backtest edge | Purge + embargo + adaptive horizon clamping |
| Sparse trigger collapse | Overfitting to micro structure | Adaptive test expansion + explicit skip reason |
| Non‑stationary drift | Masked regime shifts | Combinatorial folds increase coverage of temporal adjacency combinations |

### 7.2 Adaptive Expansion Logic (Conceptual Pseudocode)
```
test_win = base_window
while triggers_in(test_win) < min_support and test_win < cap:
    test_win = expand(test_win)
if test_win == cap and support still low:
    mark fold 'insufficient_support'
```

### 7.3 Purge & Embargo
Let test interval = [t0, t1]. We define purge window P and embargo E. Training indices T exclude:
```
[t0 - P, t1 + E]
```
This prevents subtle horizon leakage and look‑ahead in volatility clustering contexts.

### 7.4 Skip Taxonomy
| Code | Meaning |
|------|---------|
| insufficient_support | Trigger count below threshold even after expansion |
| empty_window | No overlapping data once purged |
| label_alignment_failure | Horizon mapping produced zero valid label pairs |

Skip reasons propagate into diagnostics so global robustness can be reasoned about (e.g., 60% viable folds may indicate over‑specialization).

---
## 8. Metrics, ELV, and Hart Index (Mathematical Perspective)

### 8.1 Edge Metrics
| Metric | Core Idea |
|--------|-----------|
| CRPS (negative orientation internally flipped) | Probabilistic calibration vs realized outcomes |
| PIN Quantile Edge | Contrast tail quantile predicted vs observed distribution delta |
| Information Gain | KL‑like improvement of conditional vs baseline outcome distribution |
| Edge Stability (Δ window) | Drift of cumulative edge; measures decay / overfitting onset |

### 8.2 Expected Lifecycle Value (ELV)
Not a naïve mean return; conceptually:
```
ELV = (Edge * Coverage * Persistence) * Decay_Adjustment
```
Where:
* Edge = median or tail‑robust central tendency of per‑trigger payoff.
* Coverage = fraction of evaluation horizon times triggering occurs (availability proxy).
* Persistence = 1 - (|recent_edge - long_term_edge| / denom) truncated.
* Decay_Adjustment penalizes late‑window divergence.

ELV therefore rewards *deployable* consistency, not isolated historical spikes.

### 8.3 Hart Index Composition (Weighted Multi‑Facet Score)
Facet families (conceptual, weights configurable):
1. Performance Quality (edge calibration, CRPS reliability)
2. Robustness (bootstrap p‑values, sensitivity deltas)
3. Structural Causality (transfer entropy, Granger causality, redundancy mutual information)
4. Complexity & Regime Adaptability (dfa alpha, complexity metrics if enabled)
5. Readiness & Coverage (live trigger prior, fold coverage factor)

Each sub‑metric normalized to [0,1] via robust scaling (IQR or winsorized boundaries). Missing metrics do **not** default to median; they reduce confidence potential — driving the philosophy that unmeasured facets reduce trust.

### 8.4 Causality Primitives
| Metric | Interpretation |
|--------|---------------|
| Transfer Entropy | Asymmetric information flow beyond linear correlation |
| Granger p‑value | Linear predictive precedence; low p ⇒ lead effect |
| Redundancy MI | High redundancy suggests duplicative signal set (penalized) |

---
## 9. Reproducibility & Provenance (Schema)
`reproducibility.json` (illustrative keys):
```json
{
  "timestamp": "2025-09-27T12:34:56Z",
  "git": {"commit": "abc123", "dirty": false},
  "python": "3.10.x",
  "packages": {"pandas": "x.y.z", "numpy": "..."},
  "seed": 193,
  "objectives": ["edge_crps", "edge_pin_q10", "coverage"],
  "horizons": [5, 10, 21],
  "env": {"FEATURES_PAIRWISE_MAX_TICKERS": "15"},
  "run_mode": "discover"
}
```
This file is **write‑once per run**; downstream processes treat it as immutable metadata.

---
## 10. Performance & Caching (Decomposition)

### 10.1 Pairwise Caching Contract
Key = `(spec_version, data_hash, ticker, bench)`. Changing spec logic should bump `PAIRWISE_SPEC_VERSION` to invalidate stale semantics. Data hash isolates only the *minimal necessary columns* to avoid spurious invalidations.

### 10.2 Memory Footprint Adaptive Jobs
Heuristic: If wide frame footprint exceeds threshold → clamp concurrency to preserve workstation responsiveness. This defers out‑of‑memory stalls without forcing manual intervention mid‑run.

### 10.3 BLAS Thread Pinning
Ensures multi-threaded Python layer decisions (joblib) are not drowned out by nested BLAS parallelism leading to CPU oversubscription and noisy wall time variance.

### 10.4 Failure Recovery Philosophy
If a feature spec raises an exception, the system logs `[feat warn]` with contextual name but does **not** fabricate a fallback vector. Missing data surfaces as a column absence, making the cost / effect of the failure explicit.

---
## 11. Configuration Surfaces (Extended)

### 11.1 Hierarchy
1. Code defaults in `config.py` (authoritative base)
2. Environment overrides (ephemeral run‑time tuning)
3. In‑process dynamic modifications (discouraged except for experimental notebooks)

### 11.2 Mutability Discipline
`settings` should not be mutated during GA evaluation; doing so would break deterministic seeding assumptions. Any experiment requiring mid‑evaluation shift should instead partition runs.

---
## 12. Running, Modes & Operational Patterns

| Scenario | Configuration | Expected Outcome |
|----------|---------------|------------------|
| Full discovery + validation | Default `main.py` run | Produces artifacts & diagnostics |
| Discovery only (faster) | `settings.run_mode='discover'` | Skip heavy post evaluation phases |
| Stress test pairwise | Limit tickers + set low progress heartbeat | Validate scaling safely |
| Profiling run | `FEATURES_PAIRWISE_PROFILE=1` | cProfile top 25 for pairwise block |

**Incremental Experiment Loop:**
1. Narrow universe (`FEATURES_PAIRWISE_MAX_TICKERS=8`)
2. Add / tweak feature spec
3. Run discovery only
4. Inspect Pareto + sparsity diagnostics
5. Scale universe back upward incrementally.

---
## 13. Extensibility Patterns (In Depth)

### 13.1 Adding a Feature (Single Asset)
1. Add lambda in `FEAT` with name pattern `<namespace>.<descriptor>`.
2. Use helper `_col(df, ticker, FIELD)` to avoid brittle column logic.
3. Apply transformations, then rely on global shift in registry code.

### 13.2 Adding a Pairwise Spec
```python
PAIR_SPECS["x.new_signal_42"] = lambda D,a,b,min_periods=None: your_fn(D,a,b,min_periods)
```
If `min_periods` unused, accept it in signature to keep forward compatibility (shim filters kwargs anyway).

### 13.3 Adding an Objective
1. Implement metric function returning scalar.
2. Register orientation (min/max) in `OBJECTIVE_TRANSFORMS`.
3. Append name to `settings.ga.objectives`.
4. Add unit test locking invariants (e.g., monotonic transform correctness).

### 13.4 Adding Diagnostics
Design rule: diagnostics produce *append‑only* artifacts; no modification to evaluation results. Place under `eval/diagnostics/` and hook call in validation pipeline.

### 13.5 Anti‑Patterns to Avoid
| Anti‑Pattern | Risk |
|--------------|------|
| Hidden global state mutation in feature functions | Non‑deterministic results across runs |
| Swallowing exceptions silently | Invisible data quality degradation |
| In‐place DataFrame mutation during iteration | Hard‑to‑trace side effects |
| Adding penalty defaults | Reintroduction of silent bias |

---
## 14. Testing Strategy (Extended)
| Layer | Current | Suggested Additions |
|-------|---------|---------------------|
| Objective transforms | Orientation + transform tests | Boundary tests on negative / zero inputs |
| CPCV viability | Basic split sanity | Randomized trigger density fuzz tests |
| Feature integrity | Ad‑hoc manual | Hash snapshots for critical feature subsets |
| Pairwise caching | Pending | Simulate repeated calls; assert identical hash hits |
| Hart Index components | Usage audit only | Synthetic dataset w/ controlled causality |

**Philosophy:** Tests prefer *structural invariants* (e.g., “fold count non‑increasing after applying stricter min support”) over raw number expectations which are brittle under data growth.

---
## 15. Diagnostics & Artifact Anatomy
| Artifact | Schema Highlights | Analysis Use |
|----------|------------------|--------------|
| `fold_sparsity_diagnostics.csv` | fold_id, triggers, support_ratio, skip_reason | Detect systematic sparsity bias |
| `pareto_front.csv` | individual, objective_* columns, repro header | Re‑rank under alternative weighting |
| `forecast_slate.csv` | enriched candidate metrics + scores | Allocation & deployment gating |
| `hart_index_usage.txt` | per metric non‑null counts | Identify missing robustness facets |
| `tradeable_setups.csv` | final filtered actionable setups | Downstream execution handoff |
| `reproducibility.json` | run metadata | Provenance & diffing |

---
## 16. Failure Modes & Mitigations (Expanded)
| Symptom | Root Cause Model | First Response | Longer Term Hardening |
|---------|------------------|----------------|-----------------------|
| Flood of skipped folds | Overly narrow trigger set | Lower min fires temporarily | Feature diversity / broaden predicate space |
| Memory clamp triggers frequently | Universe size vs workstation RAM | Raise warn threshold incrementally | Implement streaming pair chunking |
| Hart Index low despite high raw edge | Missing causality / redundancy metrics | Enable causality module for dataset | Expand diagnostic coverage |
| Pairwise cache churn | Spec version accidentally bumped | Revert env or version; inspect hash keys | Introduce cache hit ratio telemetry |
| NaN cascade in new feature | Missing underlying field | Instrument `_col` fallback print | Add schema validation pre‑feature stage |

---
## 17. Design Decisions (Detailed Commentary)
1. **Unified Maximize Space:** Reduces mental parser load reading GA code and prevents asymmetric error when adding new objective (no need to remember sign convention). Central transform table acts as *single change surface*.
2. **No Penalty Proxies:** Penalizing with arbitrary constants introduces *rank noise* that may shuffle frontier order. Explicit exclusion ensures comparability only among empirically measurable candidates.
3. **Cache Key Minimalism:** Hashing entire wide frame would cause unnecessary invalidations; isolating per‑pair minimal columns drastically shrinks invalidation frequency.
4. **Sector Neutral Ranking vs Global Normalization:** Global z‑scores can conflate structural variance differences; sector demeaning ensures comparability without forcing distribution equality.
5. **Event Sparsity Preservation:** Dropping sparse event columns erases signals of structural silence (e.g., regime calm vs hyperactivity) which can be predictive of volatility transitions.

---
## 18. Roadmap (Narrative)
1. **Complexity Engine Reintegration:** Reintroduce permutation entropy & DFA with micro‑batch evaluation, caching intermediate symbolic transforms.
2. **Cache Telemetry Layer:** Expose hit/miss counters; adaptive spec auto‑disabling when hit rate below threshold.
3. **Automated Fold Tuning:** Use meta‑optimization to pick adaptive expansion policy parameters based on trigger density distributions.
4. **Cross‑Run Drift Monitor:** Summarize objective distribution shifts between latest run and N historical baselines.
5. **Deployment Readiness Dashboard:** Aggregate Hart Index + ELV progression, fold viability ratios, and signal decay curves.

---
## 19. Glossary
| Term | Definition | Notes |
|------|------------|-------|
| CPCV | Combinatorial Purged Cross Validation | Prevents path leakage & adjacency bias |
| DNA | Identity of strategy candidate | Order of signals canonicalized |
| ELV | Expected Lifecycle Value | Deployment viability composite |
| Hart Index | Multi‑facet trust score (0‑100) | Penalizes missing dimensions |
| Purge | Removal of training data adjacent to test interval | Leakage guard |
| Embargo | Post‑test cooling‑off interval excluded from training | Remove horizon bleed |
| Coverage Factor | Proportion of evaluation periods with triggers | Proxy for deployable frequency |
| Redundancy MI | Shared information among signals | High ⇒ potential pruning |

---
## 20. Quick Environment Reference
```bash
# Disable pairwise (fast iteration)
export FEATURES_DISABLE_PAIRWISE=1

# Constrain pairwise universe & reinforce rolling support
export FEATURES_PAIRWISE_MAX_TICKERS=12
export PAIRWISE_MIN_PERIODS=30

# Progress & profiling aids
export FEATURES_PAIRWISE_PROGRESS_EVERY=5
export FEATURES_PAIRWISE_PROFILE=1

# Memory clamp (MB threshold + fallback jobs)
export FEATURES_PAIRWISE_WARN_MB=600
export FEATURES_PAIRWISE_MAX_JOBS_ON_WARN=4
```

---
## 21. Security / Integrity Considerations (Forward‑Looking)
While not a security platform, planned integrity checks include:
| Control | Purpose |
|---------|---------|
| Input schema hash | Detect silent upstream data structure drift |
| Feature whitelist | Prevent unreviewed feature injection in CI context |
| Repro JSON signature | Tamper evidence for archival bundles |

---
## 22. License / Attribution
Add a license (MIT / Apache 2.0 recommended) to clarify usage boundaries; absence inhibits collaboration.

---
## 23. Contribution Guide (Practical)
1. Branch naming: `feat/`, `fix/`, `diag/`, `exp/` prefixes.
2. Include a *CHANGE NOTE* in PR describing effect on reproducibility (e.g., “changes objective transform semantics – bump spec version”).
3. Add or update at least one invariant test when altering evaluation logic.
4. Avoid large multi‑concern patches; keep diff localizable.

---
## 24. Status & Doctrine
Core is stable and deterministic under current dataset scale. Immediate focus: richer introspection (cache telemetry, drift monitors) and complexity metrics reintegration under performance constraints.

> “You don’t get robustness by assuming it — you get it by instrumenting your uncertainty and refusing silent success paths.” – Project Doctrine

---
## 25. Appendix: Potential Mathematical Formalizations

### A. Objective Transform Function
Let raw objective values be `O = {o_i}` with orientation flag `dir_i ∈ {+1 (maximize), -1 (minimize)}`.
Transformed: `o'_i = dir_i * o_i`. (Optional: apply scale normalization later *only* for display, never for GA ordering unless all objectives share stable scale invariance.)

### B. Coverage Factor
`coverage = (# periods with >=1 trigger for DNA) / (total evaluation periods considered)`.

### C. Persistence Component (ELV)
`persistence = 1 - |edge_recent - edge_long| / (|edge_long| + ε)` clipped to [0,1].

### D. Redundancy Mutual Information (Conceptual)
Approximate pairwise MI across signal set S, aggregate via mean or maximum; penalize high central tendency to encourage orthogonality.

---
## 26. FAQ (Anticipated)
| Question | Answer |
|----------|--------|
| Why are some folds missing? | They were explicitly deemed infeasible (see skip reasons). Fail‑closed design prevents silent padding. |
| Can I override a missing metric with zero? | Strongly discouraged — zero is *data*; absence is *structural*. Provide a real estimator or leave absent. |
| When to bump `PAIRWISE_SPEC_VERSION`? | Any semantic change to pairwise feature definitions or their statistical meaning; not for formatting changes. |
| Why shift after transform instead of before? | Ensures transformations use full available historical window and then uniformly remove look‑ahead. |
| Can I parallelize single‑asset features further? | Yes, but memory trade‑off must be measured; chunking already mitigates pressure. |

---
## 27. Closing
This document is intentionally exhaustive. Treat it as a *living engineering manifesto*: update it whenever a core invariant, caching contract, or evaluation philosophy changes. The system’s credibility compounds only if the documentation and code co‑evolve.

---
**End of Document**
