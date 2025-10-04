import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fit_ridge_raw(X, y, alpha):
    """Standardize X for ridge, then convert coefficients back to RAW scale."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    Xz = (X - means) / stds
    mdl = Ridge(alpha=alpha, fit_intercept=True).fit(Xz, y)
    beta_raw = np.asarray(mdl.coef_, dtype=float) / stds
    intercept_raw = float(mdl.intercept_ - float(np.dot(beta_raw, means)))
    r2 = float(mdl.score(Xz, y))
    return beta_raw, intercept_raw, r2, means, stds

def predict_raw(beta_raw, intercept_raw, x):
    b = np.asarray(beta_raw, dtype=float).reshape(-1)
    xi = np.asarray(x, dtype=float).reshape(-1)
    return float(float(intercept_raw) + float(np.dot(b, xi)))

def recommend_min_norm(beta_raw, intercept_raw, x0, target, mins, maxs, allowed_idx):
    """Smallest-norm Δx on allowed indices to hit target; then clip to bounds."""
    b = np.asarray(beta_raw, dtype=float).reshape(-1)
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    mins = np.asarray(mins, dtype=float).reshape(-1)
    maxs = np.asarray(maxs, dtype=float).reshape(-1)
    allowed_idx = list(map(int, allowed_idx))

    y0 = predict_raw(b, float(intercept_raw), x0)

    bA = np.zeros_like(b)
    if len(allowed_idx) == 0:
        # nothing adjustable
        return x0.copy(), y0, y0, np.zeros_like(x0), np.zeros_like(x0, dtype=bool)
    bA[allowed_idx] = b[allowed_idx]
    norm2 = float(np.dot(bA, bA))

    if norm2 < 1e-12:
        return x0.copy(), y0, y0, np.zeros_like(x0), np.zeros_like(x0, dtype=bool)

    deltas = bA * ((float(target) - y0) / norm2)
    x_rec = x0.copy()
    clipped = np.zeros_like(x0, dtype=bool)

    for j in allowed_idx:
        cand = x0[j] + deltas[j]
        clp = float(np.clip(cand, mins[j], maxs[j]))
        if clp != cand:
            clipped[j] = True
        deltas[j] = clp - x0[j]
        x_rec[j] = x0[j] + deltas[j]

    y_rec = predict_raw(b, float(intercept_raw), x_rec)
    return x_rec, y0, y_rec, deltas, clipped
