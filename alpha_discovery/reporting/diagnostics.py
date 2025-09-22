from __future__ import annotations
from typing import List
import os
import pandas as pd
from datetime import datetime

from ..config import settings
from ..core.splits import HybridSplits


def write_split_audit(splits: HybridSplits, output_dir: str) -> None:
    """
    Writes a CSV artifact detailing the exact boundaries and properties of all
    generated splits for a run, aiding in QA and reproducibility.

    Args:
        splits: The HybridSplits object containing all split data.
        output_dir: The base directory for the run where artifacts are saved.
    """
    audit_data = []
    purge_days = settings.validation.purge_days
    embargo_days = settings.validation.embargo_days

    # 1. Process Discovery CV splits
    for i, (train_idx, test_idx) in enumerate(splits.discovery_cv, 1):
        audit_data.append({
            "Fold": i,
            "Type": "CV Train",
            "Start Date": train_idx.min().date(),
            "End Date": train_idx.max().date(),
            "n_days": len(train_idx),
            "Purge": purge_days,
            "Embargo": embargo_days
        })
        audit_data.append({
            "Fold": i,
            "Type": "CV Test",
            "Start Date": test_idx.min().date(),
            "End Date": test_idx.max().date(),
            "n_days": len(test_idx),
            "Purge": None,
            "Embargo": None
        })

    # 2. Process True OOS splits
    for i, oos_idx in enumerate(splits.oos, 1):
        audit_data.append({
            "Fold": i,
            "Type": "OOS",
            "Start Date": oos_idx.min().date(),
            "End Date": oos_idx.max().date(),
            "n_days": len(oos_idx),
            "Purge": None,
            "Embargo": None
        })
        
    # 3. Process Forward Gauntlet
    if splits.gauntlet is not None and not splits.gauntlet.empty:
        audit_data.append({
            "Fold": 1,
            "Type": "Gauntlet",
            "Start Date": splits.gauntlet.min().date(),
            "End Date": splits.gauntlet.max().date(),
            "n_days": len(splits.gauntlet),
            "Purge": None,
            "Embargo": None
        })
        
    if not audit_data:
        return

    audit_df = pd.DataFrame(audit_data)
    
    diag_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d")
    file_path = os.path.join(diag_dir, f"split_audit_{ts}.csv")
    audit_df.to_csv(file_path, index=False)
    print(f"Split audit saved to: {file_path}")
