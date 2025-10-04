"""Utilities for canonical trade fingerprints and ledger de-duplication."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd


def canonical_signals_fingerprint(
    signals: Sequence[str],
    hits: Optional[Mapping[str, Any]] = None,
    *,
    mode: str = "soft_and",
    window_days: Optional[int] = None,
) -> str:
    """Return deterministic fingerprint for a set of signals and their hits."""

    sorted_signals = [str(sig) for sig in sorted(signals)]
    signals_part = ",".join(sorted_signals)

    hits_part = ""
    if hits:
        ordered_hits = []
        for sig in sorted_signals:
            flag = bool(hits.get(sig))
            ordered_hits.append(f"{sig}:{'1' if flag else '0'}")
        hits_part = f"|hits=[{','.join(ordered_hits)}]"

    window_part = f"|window={int(window_days)}d" if window_days is not None else ""

    return f"mode={mode}|signals=[{signals_part}]{hits_part}{window_part}"


def _relative_strike(strike: Any, underlying: Any) -> Optional[float]:
    try:
        strike_val = float(strike)
        underlying_val = float(underlying)
    except (TypeError, ValueError):
        return None
    if underlying_val == 0:
        return None
    return (strike_val / underlying_val) - 1.0


def options_structure_key(trade: Mapping[str, Any]) -> str:
    """Build deterministic option structure descriptor for a ledger row."""

    option_type = str(trade.get("option_type", "unknown")).lower()
    horizon = trade.get("horizon_days")
    if horizon is None:
        horizon = trade.get("holding_days_actual")
    try:
        horizon_int = int(float(horizon))
    except (TypeError, ValueError):
        horizon_int = None
    tenor_part = f"tenor={horizon_int}d" if horizon_int is not None else "tenor=NA"

    rel_strike = _relative_strike(trade.get("strike"), trade.get("entry_underlying"))
    if rel_strike is None:
        strike_part = f"strike={trade.get('strike')}"
    else:
        strike_part = f"strike_rel={rel_strike:.4f}"

    delta = trade.get("delta_target")
    if delta is None:
        delta = trade.get("delta_achieved")
    try:
        delta_part = f"delta={float(delta):.3f}"
    except (TypeError, ValueError):
        delta_part = "delta=NA"

    width = trade.get("spread_width")
    try:
        width_part = f"width={float(width):.3f}"
    except (TypeError, ValueError):
        width_part = ""

    components = [f"type={option_type}", tenor_part, strike_part, delta_part]
    if width_part:
        components.append(width_part)

    return ";".join(components)


def trade_uniq_key(
    *,
    setup_id: Any,
    ticker: Any,
    direction: Any,
    entry_date: Any,
    signals_fingerprint: str,
    horizon_tag: str,
    exit_policy_tag: str,
    structure_key: str,
) -> str:
    """Canonical hash identifying a trade instance across the pipeline."""

    try:
        entry_ts = pd.Timestamp(entry_date).normalize()
        entry_part = entry_ts.isoformat()
    except Exception:
        entry_part = str(entry_date)

    payload_parts = [
        str(setup_id or ""),
        str(ticker or ""),
        str(direction or ""),
        entry_part,
        signals_fingerprint,
        str(horizon_tag or ""),
        str(exit_policy_tag or ""),
        structure_key,
    ]
    payload = "|".join(payload_parts)
    digest = hashlib.sha256(payload.encode("utf-8", "ignore")).hexdigest()
    return digest[:16]


def _horizon_tag(value: Any) -> str:
    try:
        val = int(float(value))
    except (TypeError, ValueError):
        return "H?"
    return f"H{val}"


def dedupe_trade_ledger(
    ledger: pd.DataFrame,
    *,
    setup_id: str,
    ticker: str,
    direction: str,
    signals_fingerprint: str,
    exit_policy_tag: str,
    allow_pyramiding: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Attach uniq_key to each row and collapse duplicates deterministically."""

    if ledger is None or ledger.empty:
        return ledger, {"n_dups_suppressed": 0, "n_dups_merged": 0}

    df = ledger.copy()
    df['trigger_date'] = pd.to_datetime(df.get('trigger_date'), errors='coerce')
    df['options_structure_key'] = df.apply(options_structure_key, axis=1)

    def _row_key(row: pd.Series) -> str:
        horizon_lbl = _horizon_tag(row.get('horizon_days'))
        return trade_uniq_key(
            setup_id=setup_id,
            ticker=ticker,
            direction=direction,
            entry_date=row.get('trigger_date'),
            signals_fingerprint=signals_fingerprint,
            horizon_tag=horizon_lbl,
            exit_policy_tag=row.get('exit_policy_id') or exit_policy_tag,
            structure_key=row.get('options_structure_key', ''),
        )

    df['uniq_key'] = df.apply(_row_key, axis=1)

    group_sizes = df.groupby('uniq_key')['uniq_key'].transform('size')
    df['merge_count'] = group_sizes - 1

    if allow_pyramiding:
        # Merge duplicate entries by summing size-like columns while retaining first row semantics.
        agg_cols: Dict[str, Any] = {}
        for col in ['contracts', 'capital_allocated', 'capital_allocated_used', 'unrealized_pnl', 'realized_pnl', 'pnl_dollars']:
            if col in df.columns:
                agg_cols[col] = 'sum'
        base = df.groupby('uniq_key', sort=False).first()
        sums = df.groupby('uniq_key').agg(agg_cols) if agg_cols else pd.DataFrame(index=base.index)
        for col, series in sums.items():
            base[col] = series
        base['merge_count'] = df.groupby('uniq_key')['merge_count'].max()
        deduped = base.reset_index(drop=False)
    else:
        keep_mask = ~df.duplicated('uniq_key', keep='first')
        deduped = df.loc[keep_mask].copy()

    n_suppressed = int(len(df) - len(deduped))
    n_merged = int((deduped.get('merge_count', 0) > 0).sum())

    stats = {
        "n_dups_suppressed": n_suppressed,
        "n_dups_merged": n_merged,
    }

    return deduped, stats
