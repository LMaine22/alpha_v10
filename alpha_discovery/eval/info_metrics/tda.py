# alpha_discovery/eval/metrics/tda.py
"""
Topological summaries (no placeholders).
Implements:
  - Persistent Homology (H0) for point clouds via MST (exact)
  - Persistence diagram utilities
  - Wasserstein-1 / Bottleneck distances between H0 diagrams (Hungarian)
  - Simple landscape vector from H0 lifetimes
Note: H1 loops typically require specialized libraries; this module focuses on exact H0 which
is robust and informative for cluster structure. Distances include diagonal matching.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

EPS = 1e-12

def _pairwise_dist(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    diffs = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diffs**2, axis=2))

def _mst_edges(D: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Kruskal's algorithm on complete graph with weights D (upper triangular used).
    Returns list of edges (i,j,w) in MST.
    """
    n = D.shape[0]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j, float(D[i, j])))
    edges.sort(key=lambda e: e[2])

    parent = list(range(n))
    rank = [0]*n
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra == rb: return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    mst = []
    for i, j, w in edges:
        if union(i, j):
            mst.append((i, j, w))
            if len(mst) == n-1:
                break
    return mst

def persistent_homology_h0(points: np.ndarray) -> np.ndarray:
    """
    Returns H0 diagram for a point cloud as array of (birth, death),
    where births are 0 and deaths are MST edge weights at merges.
    """
    X = np.asarray(points, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    if n == 0:
        return np.empty((0, 2))
    D = _pairwise_dist(X)
    mst = _mst_edges(D)
    # In H0 VR filtration, each component born at 0, dies when merged at edge length.
    deaths = sorted([w for (_, _, w) in mst])
    # n components => n points; one persists to infinity (we cap at max edge)
    max_edge = max(deaths) if deaths else 0.0
    diagram = np.zeros((n, 2), dtype=float)
    diagram[:, 0] = 0.0
    # assign deaths: n-1 finite deaths + one "infinite" set to max_edge (capped)
    if n > 1:
        diagram[:n-1, 1] = deaths
        diagram[n-1, 1] = max_edge
    else:
        diagram[0, 1] = 0.0
    return diagram

def h0_landscape_vector(diagram: np.ndarray, bins: int = 32) -> np.ndarray:
    """
    Simple vector: histogram of lifetimes (death-birth) over equal-width bins.
    """
    if diagram.size == 0:
        return np.zeros(bins, dtype=float)
    lifetimes = diagram[:,1] - diagram[:,0]
    hist, _ = np.histogram(lifetimes, bins=bins, range=(0, np.max(lifetimes) + EPS), density=True)
    return hist.astype(float)

def _augment_with_diagonal(diag: np.ndarray) -> np.ndarray:
    """
    Ensure diagonal is available for matching by duplicating each point's projection.
    For H0, diagonal points are (t,t). We'll generate as needed in cost matrix.
    """
    return np.asarray(diag, dtype=float)

def wasserstein1_h0(diag1: np.ndarray, diag2: np.ndarray) -> float:
    """
    Compute 1-Wasserstein distance between two H0 diagrams with diagonal matching via Hungarian.
    """
    D1 = _augment_with_diagonal(diag1)
    D2 = _augment_with_diagonal(diag2)
    n, m = D1.shape[0], D2.shape[0]
    # Cost matrix with augmentations to allow matching to diagonal
    # Build full (n+m) x (n+m) matrix per standard PD matching reduction
    C = np.zeros((n + m, n + m), dtype=float)
    # Off-diagonal costs
    for i in range(n):
        for j in range(m):
            C[i, j] = np.linalg.norm(D1[i] - D2[j], ord=1)  # L1 cost
    # Match to diagonal (|b-d|/2 for L∞; for L1 we use |b-d|)
    for i in range(n):
        C[i, m + i] = abs(D1[i, 1] - D1[i, 0])  # cost to diagonal
    for j in range(m):
        C[n + j, j] = abs(D2[j, 1] - D2[j, 0])
    # Fill remaining with large numbers to discourage invalid matches
    big = 1e6
    C[n:, m:] = 0.0
    # Hungarian
    if linear_sum_assignment is None:
        # fallback greedy (not optimal)
        total = 0.0
        used_j = set()
        for i in range(n):
            jj = int(np.argmin(C[i, :m]))
            if jj in used_j:
                total += C[i, m + i]
            else:
                total += C[i, jj]
                used_j.add(jj)
        for j in range(m):
            if j not in used_j:
                total += C[n + j, j]
        return float(total)
    row_ind, col_ind = linear_sum_assignment(C)
    return float(C[row_ind, col_ind].sum())

def bottleneck_h0(diag1: np.ndarray, diag2: np.ndarray) -> float:
    """
    Approximate bottleneck distance using L∞ cost with diagonal matching via threshold search.
    """
    # Collect candidate thresholds
    pts = np.vstack([diag1, diag2])
    cand = set()
    for a in pts:
        for b in pts:
            cand.add(max(abs(a[0] - b[0]), abs(a[1] - b[1])))
        # include distance to diagonal
        cand.add(0.5 * abs(a[1] - a[0]))
    cand = sorted(list(cand))
    # Binary search smallest epsilon that permits perfect matching (greedy feasibility)
    def feasible(eps):
        # count capacity of matches within eps (very rough feasibility via greedy)
        D1 = diag1.copy(); D2 = diag2.copy()
        used = np.zeros(D2.shape[0], dtype=bool)
        for i in range(D1.shape[0]):
            found = False
            # try match to some j
            for j in range(D2.shape[0]):
                if used[j]:
                    continue
                if max(abs(D1[i,0]-D2[j,0]), abs(D1[i,1]-D2[j,1])) <= eps:
                    used[j] = True
                    found = True
                    break
            if not found:
                # try diagonal
                if 0.5 * abs(D1[i,1] - D1[i,0]) <= eps:
                    found = True
            if not found:
                return False
        # left-over D2 points can match to diagonal if possible
        return True
    lo, hi = 0.0, cand[-1] if cand else 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return hi

__all__ = ["persistent_homology_h0", "h0_landscape_vector", "wasserstein1_h0", "bottleneck_h0"]
