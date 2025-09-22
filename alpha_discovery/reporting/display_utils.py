# alpha_discovery/reporting/display_utils.py
"""
Shared utilities for formatting and display in reporting modules.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterable, Union


def build_signal_meta_map(signals_metadata: Optional[Iterable[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a metadata lookup map keyed by multiple possible identifiers for robustness.
    """
    meta_map: Dict[str, Dict[str, Any]] = {}
    if not signals_metadata:
        return meta_map
    for m in signals_metadata:
        if not isinstance(m, dict):
            continue
        for key_field in ("signal_id", "id", "name"):
            v = m.get(key_field)
            if v is not None:
                meta_map[str(v)] = m
    return meta_map


def extract_signal_id_token(item: Any) -> str:
    """
    Given a setup item (string or dict), return the best-guess identifier string.
    """
    if isinstance(item, dict):
        for k in ("signal_id", "id", "name", "label"):
            if item.get(k) is not None:
                return str(item[k])
        return str(item)
    return str(item)


def desc_from_meta(setup_items: Any, meta_map: Dict[str, Dict[str, Any]]) -> str:
    """
    Build a one-line description from signal metadata if available.
    Falls back to joining best-guess IDs if no labels are found.
    Accepts: a single item, a list/tuple of items, or None.
    Each item may be a dict with keys like label/name/id/signal_id and optional op/operator.
    """
    # Normalize to an iterable
    if isinstance(setup_items, (list, tuple)):
        it = list(setup_items)
    elif setup_items is None:
        it = []
    else:
        it = [setup_items]

    parts: List[str] = []
    for item in it:
        # Extract an identifier token to look up in the metadata map
        token = extract_signal_id_token(item)
        meta = meta_map.get(token) or meta_map.get(str(token))
        
        # Prioritize the pre-computed 'description' field from the compiler.
        if meta and meta.get("description"):
            parts.append(str(meta["description"]).strip())
            continue

        if meta:
            label = meta.get("label") or meta.get("name") or meta.get("id") or str(token)
            # Prefer item-level operator if present; else use metadata operator if present
            op = None
            if isinstance(item, dict):
                op = item.get("op") or item.get("operator")
            if op is None:
                op = meta.get("op") or meta.get("operator")
            parts.append(f"{label} {op}".strip() if op else str(label))
        else:
            # No metadata: fall back to the token itself (and any inline operator)
            if isinstance(item, dict):
                label = item.get("label") or item.get("name") or item.get("id") or token
                op = item.get("op") or item.get("operator")
                parts.append(f"{label} {op}".strip() if op else str(label))
            else:
                parts.append(str(token))

    return " AND ".join(p for p in parts if p)


def format_setup_description(sol: Dict[str, Any]) -> str:
    """
    Fallback: derive a one-line description from `individual`/`setup`/`signals` fields.
    Accepts a solution dict with possible keys:
      - "individual": Tuple[str, List[Union[str, dict]]]  (ticker, signals)
      - "setup":      List[Union[str, dict]]
      - "signals":    List[Union[str, dict]]
    """
    desc = (sol.get("description") or "").strip() if isinstance(sol, dict) else ""
    if desc:
        return desc

    setup_items = None
    if isinstance(sol, dict):
        setup_items = sol.get("individual", sol.get("setup"))
        if isinstance(setup_items, tuple) and len(setup_items) == 2:
            _, setup_items = setup_items
        if not setup_items:
            setup_items = sol.get("signals") or []

    parts: List[str] = []
    for item in (setup_items or []):
        if isinstance(item, dict):
            label = item.get("label") or item.get("name") or item.get("id") or str(item)
            op = item.get("op") or item.get("operator")
            parts.append(f"{label} {op}".strip() if op else str(label))
        else:
            parts.append(str(item))
    return " AND ".join(p for p in parts if p)


def signal_ids_str(setup_items: Any) -> str:
    """
    Canonical comma-separated list of signal identifiers for CSV.
    """
    if isinstance(setup_items, (list, tuple)):
        # Sort signals to match setup_id ordering for consistency
        signal_ids = [extract_signal_id_token(x) for x in setup_items]
        return ", ".join(sorted(signal_ids))
    if setup_items is None:
        return ""
    return extract_signal_id_token(setup_items)
