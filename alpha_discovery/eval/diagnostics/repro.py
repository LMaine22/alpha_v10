"""Reproducibility metadata utilities.

Purpose
-------
Stamp each run with a minimal, machine & code provenance header so that
results can be audited / reproduced later. This intentionally avoids
importing heavy internal modules to remain side‑effect free.
"""
from __future__ import annotations
import os
import sys
import json
import platform
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional


def _git_commit() -> str:
    """Return current git HEAD commit hash (empty string if unavailable)."""
    try:
        out = subprocess.check_output([
            "git", "rev-parse", "HEAD"
        ], stderr=subprocess.DEVNULL, timeout=2)
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def _pkg_versions() -> Dict[str, str]:
    """Collect light-weight core package versions (best effort)."""
    vers: Dict[str, str] = {}
    try:
        import numpy as _np  # type: ignore
        vers["numpy"] = _np.__version__
    except Exception:
        pass
    try:
        import pandas as _pd  # type: ignore
        vers["pandas"] = _pd.__version__
    except Exception:
        pass
    try:
        import joblib as _joblib  # type: ignore
        vers["joblib"] = _joblib.__version__
    except Exception:
        pass
    return vers


def reproducibility_header(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a provenance header dictionary.

    Parameters
    ----------
    extra : optional dict
        Additional user-provided key/values to merge (shallow update).
    """
    hdr: Dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "executable": sys.executable,
        "env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS", ""),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", ""),
        },
        "packages": _pkg_versions(),
    }
    if extra:
        hdr.update(extra)
    return hdr


def write_repro_json(path: str, extra: Optional[Dict[str, Any]] = None) -> str:
    """Write provenance header JSON to path and return the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(reproducibility_header(extra), f, indent=2, sort_keys=True)
    return path

__all__ = ["reproducibility_header", "write_repro_json"]
