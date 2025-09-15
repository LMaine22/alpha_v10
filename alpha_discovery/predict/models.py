from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV

def make_classifier(spec: Dict[str, Any]):
    t = spec.get("type", "logistic")
    params = spec.get("params", {})
    calibrate = bool(spec.get("calibrate", True))
    if t == "logistic":
        base = LogisticRegression(**params)
    else:
        raise ValueError(f"Unsupported classifier type: {t}")
    if calibrate:
        # isotonic is slower; 'sigmoid' (Platt) is fine here
        return CalibratedClassifierCV(base, cv=spec.get("cv_folds", 4), method="sigmoid")
    return base

def make_regressor(spec: Dict[str, Any]) -> GradientBoostingRegressor:
    t = spec.get("type", "gbr")
    params = spec.get("params", {})
    if t == "gbr":
        return GradientBoostingRegressor(**params)
    if t == "gbr_quantile":
        return GradientBoostingRegressor(**params, loss="quantile")
    raise ValueError(f"Unsupported regressor type: {t}")
