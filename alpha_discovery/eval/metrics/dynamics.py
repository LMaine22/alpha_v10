# alpha_discovery/eval/metrics/dynamics.py
"""
Dynamical systems metrics for forecast evaluation (no placeholders).
"""
from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as pd

def _as_1d_series(series: pd.Series) -> np.ndarray:
    if isinstance(series, pd.Series):
        x = series.values.astype(float)
    else:
        x = np.asarray(series, dtype=float)
    return x.ravel()

def dfa_alpha(series: pd.Series, win_sizes: List[int] = None) -> float:
    """
    Detrended Fluctuation Analysis (Peng et al.):
    1) Integrate series (subtract mean, cumulative sum).
    2) For each window size n, divide profile into non-overlapping segments,
       detrend each segment by linear fit, compute RMS fluctuation F(n).
    3) Fit log F(n) ~ a + alpha * log n. Return slope alpha.
    """
    x = _as_1d_series(series)
    x = x[np.isfinite(x)]
    N = x.size
    if N < 8:
        return np.nan
    profile = np.cumsum(x - x.mean())

    if win_sizes is None:
        # log-spaced windows between 4 and N//4
        nmin = 4
        nmax = max(8, N // 4)
        k = int(np.clip(np.log2(N), 6, 20))
        win_sizes = np.unique(np.floor(np.logspace(np.log10(nmin), np.log10(nmax), k)).astype(int))
    Fs = []
    ns = []
    for n in win_sizes:
        if n < 2 or n > N//2:
            continue
        m = N // n
        if m < 2:
            continue
        # Use first m*n points
        prof = profile[:m*n].reshape(m, n)
        t = np.arange(n)
        # Detrend each segment
        # linear fit y ~ a*t + b
        tt = t - t.mean()
        denom = np.sum(tt*tt)
        if denom == 0:
            continue
        seg_F = []
        for seg in prof:
            a = np.sum(tt * (seg - seg.mean())) / denom
            trend = a * tt + seg.mean()
            detr = seg - trend
            seg_F.append(np.sqrt(np.mean(detr**2)))
        F_n = np.sqrt(np.mean(np.array(seg_F)**2))
        if np.isfinite(F_n) and F_n > 0:
            Fs.append(np.log(F_n))
            ns.append(np.log(n))
    if len(ns) < 2:
        return np.nan
    # Linear regression slope
    ns = np.array(ns); Fs = np.array(Fs)
    A = np.vstack([ns, np.ones_like(ns)]).T
    alpha, _ = np.linalg.lstsq(A, Fs, rcond=None)[0]
    return float(alpha)

def rqa_metrics(series: pd.Series,
                embedding_dim: int = 3,
                delay: int = 1,
                threshold: float = 0.1) -> Dict[str, float]:
    """
    Basic Recurrence Quantification Analysis on a scalar series.

    Steps:
    - Embed time-delay vectors (Takens).
    - Build recurrence matrix R[i,j] = 1[ ||x_i - x_j|| <= eps ] where eps chosen from percentile of distances.
    - Extract:
        RR: recurrence rate
        DET: determinism = fraction of recurrence points forming diagonal lines (length>=2)
        LAM: laminarity = fraction forming vertical lines (length>=2)
        TT: trapping time = average vertical line length (>=2)
        Lmax: maximum diagonal line length
    """
    x = _as_1d_series(series)
    x = x[np.isfinite(x)]
    N = x.size
    m = int(embedding_dim)
    tau = int(delay)
    if N < (m-1)*tau + 5:
        return {k: np.nan for k in ["recurrence_rate","determinism","laminarity","trapping_time","max_line_length"]}

    # Embed
    M = N - (m-1)*tau
    X = np.empty((M, m), dtype=float)
    for i in range(m):
        X[:, i] = x[i*tau:i*tau+M]

    # Distance matrix (Chebyshev)
    diff = X[:, None, :] - X[None, :, :]
    D = np.max(np.abs(diff), axis=2)

    # Choose epsilon by threshold percentile if threshold in (0,1]; else treat as absolute
    if 0 < threshold <= 1:
        eps = np.quantile(D[np.triu_indices(M, 1)], threshold)
    else:
        eps = float(threshold)

    R = (D <= eps).astype(np.uint8)
    np.fill_diagonal(R, 0)  # remove self-recurrences for classic RQA

    # Recurrence Rate
    RR = R.sum() / (M*M - M)

    # Helper to compute line lengths along a given axis
    def _line_lengths(mat: np.ndarray, min_len: int = 2, axis: int = 1):
        # count contiguous ones along rows (axis=1) or columns (axis=0)
        if axis == 1:
            A = mat
        else:
            A = mat.T
        lengths = []
        for row in A:
            c = 0
            for v in row:
                if v:
                    c += 1
                elif c > 0:
                    if c >= min_len:
                        lengths.append(c)
                    c = 0
            if c >= min_len:
                lengths.append(c)
        return lengths

    # Diagonal lines: scan diagonals of R
    diag_lengths = []
    for k in range(-M+1, M):
        diag = np.diag(R, k=k)
        c = 0
        for v in diag:
            if v:
                c += 1
            elif c > 0:
                if c >= 2:
                    diag_lengths.append(c)
                c = 0
        if c >= 2:
            diag_lengths.append(c)

    vert_lengths = _line_lengths(R, min_len=2, axis=0)

    if len(diag_lengths) == 0:
        DET = 0.0
        Lmax = 0
    else:
        det_points = sum(l for l in diag_lengths)
        total_points = R.sum()
        DET = det_points / total_points if total_points > 0 else 0.0
        Lmax = max(diag_lengths)

    if len(vert_lengths) == 0:
        LAM = 0.0
        TT = 0.0
    else:
        lam_points = sum(l for l in vert_lengths)
        total_points = R.sum()
        LAM = lam_points / total_points if total_points > 0 else 0.0
        TT = float(np.mean(vert_lengths))

    return {
        "recurrence_rate": float(RR),
        "determinism": float(DET),
        "laminarity": float(LAM),
        "trapping_time": float(TT),
        "max_line_length": int(Lmax),
    }

__all__ = ["dfa_alpha", "rqa_metrics"]
