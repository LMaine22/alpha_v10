"""Utilities for deterministic GA evaluation plans over nested walk-forward folds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import hashlib
import pandas as pd


def _index_signature(idx: Optional[pd.Index]) -> str:
    """Return a compact signature for a datetime index (start/end/count)."""
    if idx is None or len(idx) == 0:
        return "empty"
    start = pd.Timestamp(idx[0]).isoformat()
    end = pd.Timestamp(idx[-1]).isoformat()
    return f"{start}:{end}:{len(idx)}"


@dataclass(frozen=True)
class InnerFoldPlan:
    """One NPWF inner fold with explicit train/test indices."""

    fold_id: str
    train_idx: pd.Index
    test_idx: pd.Index

    def signature(self) -> str:
        """Deterministic signature for hashing and logging."""
        return f"{self.fold_id}:{_index_signature(self.train_idx)}:{_index_signature(self.test_idx)}"


@dataclass(frozen=True)
class GADataSpec:
    """Bundle of outer PAWF window with associated NPWF inner folds."""

    outer_id: str
    train_idx: pd.Index
    test_idx: pd.Index
    inner_folds: List[InnerFoldPlan] = field(default_factory=list)
    label_horizon: int = 0
    embargo_days: int = 0
    metadata: Optional[dict] = None

    @property
    def fold_hash(self) -> str:
        """Deterministic hash covering outer window and inner fold boundaries."""
        payload_parts = [
            self.outer_id,
            _index_signature(self.train_idx),
            _index_signature(self.test_idx),
        ]
        payload_parts.extend(f.signature() for f in self.inner_folds)
        payload = "|".join(payload_parts)
        return hashlib.sha256(payload.encode("utf-8", "ignore")).hexdigest()[:16]

    @property
    def n_folds(self) -> int:
        return len(self.inner_folds)

    def summary(self) -> dict:
        """Lightweight summary for logging or artifacts."""
        return {
            "outer_id": self.outer_id,
            "train_span": _index_signature(self.train_idx),
            "test_span": _index_signature(self.test_idx),
            "n_inner_folds": len(self.inner_folds),
            "fold_hash": self.fold_hash,
        }
