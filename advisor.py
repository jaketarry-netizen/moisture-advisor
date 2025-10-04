# advisor.py  — Streamlit app using your original helpers

import streamlit as st

# ---------- FAST BOOT UI ----------
st.set_page_config(page_title="Moisture Advisor", layout="wide")
st.title("Moisture Advisor")
st.caption("Upload your wide Excel/CSV, train a ridge model, and get setpoint recommendations to hit a target moisture.")

# Sidebar controls shown early
uploaded = st.file_uploader("Upload Excel (.xlsx) or CSV", type=["xlsx", "xls", "csv"])
alpha = st.sidebar.slider("Ridge λ (regularization)", 0.0, 100.0, 1.0, 0.5)
target = st.sidebar.number_input("Target moisture", value=8.70, step=0.01, format="%.2f")

# Bail out early until we have a file
if not uploaded:
    st.info("Waiting for file upload…")
    st.stop()

# ---------- HEAVY IMPORTS (after we have a file) ----------
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# ---------- YOUR ORIGINAL HELPERS (unchanged) ----------
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

# ---------- LOAD DATA ----------
@st.cache_data(show_spinner=False)
def read_any(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

df = read_any(uploaded)
st.write(f"Loaded **{len(df)} rows** × **{df.shape[1]} columns**")

# ---------- COLUMN PICKERS ----------
DEFAULT_FEATS = [
    "Value_Rate", "Density",
    "Value_PreWater","Value_PreSteam","Value_ExtruderW","Value_ExtruderS",
    "Value_Zone1","Value_Zone2","Value_Zone3","Value_Zone4",
    "Value_RetentionT","Value_RetentionM","Value_RetentionB",
]

num_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]
guess = "Moisture" if "Moisture" in df.columns else ("Moisture Ohaus" if "Moisture Ohaus" in df.columns else (num_cols[0] if num_cols else None))
moist_col = st.selectbox("Moisture column", options=num_cols, index=(num_cols.index(guess) if guess in num_cols else 0))

feat_options = [c for c in DEFAULT_FEATS if c in df.columns]
features = st.multiselect("Adjustable features (knobs)", options=feat_options, default=feat_options)

# ---------- TRAIN ----------
df2 = coerce_numeric(df.copy(), [moist_col] + features)
train = df2[[moist_col] + features].dropna()
if len(train) < max(8, len(features) + 2):
    st.warning("Not enough complete rows to fit. Try fewer features or clean missing values.")
    st.stop()

X = train[features].values
y = train[moist_col].values
beta_raw, intercept, r2, means, stds = fit_ridge_raw(X, y, alpha)
mins = np.nanmin(X, axis=0); maxs = np.nanmax(X, axis=0)
st.markdown(f"**Model R²:** {r2:.3f}")

# ---------- CURRENT SETTINGS INPUT ----------
st.subheader("Enter current settings")
x0 = np.nanmedian(X, axis=0).astype(float)  # medians as starting point
cols = st.columns(3)
allowed_idx = []
for i, f in enumerate(features):
    with cols[i % 3]:
        pad = (maxs[i] - mins[i]) * 0.02
        x0[i] = st.number_input(
            f"{f}  [{mins[i]:.2f}…{maxs[i]:.2f}]",
            value=float(np.clip(x0[i], mins[i], maxs[i])),
            step=0.1, format="%.3f",
            min_value=float(mins[i]-pad), max_value=float(maxs[i]+pad),
        )
        if st.checkbox(f"Allow {f}", value=True, key=f"allow_{f}"):
            allowed_idx.append(i)

# ---------- RECOMMENDATION ----------
x_rec, y0, y_rec, deltas, clipped = recommend_min_norm(
    beta_raw, intercept, x0, target, mins, maxs, allowed_idx
)

st.markdown(f"**Predicted now:** {y0:.3f} → **After changes:** {y_rec:.3f} (target {target:.3f})")

# Table of changes
rows = []
for j, f in enumerate(features):
    if abs(deltas[j]) < 1e-9:
        continue
    rows.append({
        "Feature": f,
        "Current": x0[j],
        "Δ": deltas[j],
        "New": x_rec[j],
        "At bound?": "Yes" if clipped[j] else "",
    })
out = pd.DataFrame(rows)
st.subheader("Recommendation (minimal total change within observed bounds)")
st.dataframe(out.style.format({"Current":"{:.3f}","Δ":"{:+.3f}","New":"{:.3f}"}), use_container_width=True)

st.download_button(
    "Download recommendations (CSV)",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="moisture_recommendations.csv",
    mime="text/csv",
)
