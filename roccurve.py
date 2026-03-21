"""
XGBoost ROC Curve — Standalone Script
Loads the trained pipeline from ckd_best_model.pkl
and plots the ROC curve for XGBoost only.

Run from your project root:
    python plot_xgboost_roc.py
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ucimlrepo           import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics         import roc_curve, auc
from sklearn.preprocessing   import LabelEncoder

# ── 1. Load saved pipelines ──────────────────────────────────────
import os, json

pipelines = {}

# Load all pkl files in current directory that match model names
MODEL_FILES = {
    "XGBoost":      "ckd_xgboost.pkl",
    # If you only have the best model pkl, use this fallback:
}

# Primary: try loading individually saved models
for name, path in MODEL_FILES.items():
    if os.path.exists(path):
        with open(path, "rb") as f:
            pipelines[name] = pickle.load(f)
        print(f"[OK] Loaded {name} from {path}")

# Fallback: load ckd_best_model.pkl if it's XGBoost
if not pipelines and os.path.exists("ckd_best_model.pkl"):
    with open("ckd_best_model.pkl", "rb") as f:
        best_pipe = pickle.load(f)
    # Check which model is inside
    mdl = best_pipe.named_steps["model"]
    name = type(mdl).__name__
    print(f"[OK] Loaded best model: {name}")
    pipelines[name] = best_pipe

if not pipelines:
    print("[ERROR] No model pkl found. Run train_model.py first.")
    exit(1)

# ── 2. Reload & prepare test data (same split as training) ───────
print("Loading UCI CKD dataset...")
ckd = fetch_ucirepo(id=336)
df  = pd.concat([ckd.data.features, ckd.data.targets], axis=1)
df.replace("?", np.nan, inplace=True)
df.columns = [
    "age","blood_pressure","specific_gravity","albumin","sugar",
    "red_blood_cells","pus_cell","pus_cell_clumps","bacteria",
    "blood_glucose_random","blood_urea","serum_creatinine","sodium",
    "potassium","haemoglobin","packed_cell_volume",
    "white_blood_cell_count","red_blood_cell_count",
    "hypertension","diabetes_mellitus","coronary_artery_disease",
    "appetite","peda_edema","aanemia","class"
]
df["class"] = df["class"].str.strip().str.lower()
df["class"].replace("ckd\t","ckd", inplace=True)
df["class"] = df["class"].map({"ckd":1,"notckd":0})

# dirty value fixes
df["diabetes_mellitus"].replace({"\tno":"no","\tyes":"yes"," yes":"yes"}, inplace=True)
df["coronary_artery_disease"].replace("\tno","no", inplace=True)

# numeric fixes
for col in ["packed_cell_volume","white_blood_cell_count","red_blood_cell_count"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Use the same column names the model was trained on
# Try descriptive names first, fall back to short names if needed
X_full = df.drop("class", axis=1)
y      = df["class"]

# Check what names the pipeline expects
try:
    expected = list(list(pipelines.values())[0]
                    .named_steps["imputer"].feature_names_in_)
    print(f"[OK] Model expects: {expected[:5]}...")

    # Map descriptive → short if model used short names
    DESC_TO_SHORT = {
        "blood_pressure":"bp", "specific_gravity":"sg", "albumin":"al",
        "sugar":"su", "red_blood_cells":"rbc", "pus_cell":"pc",
        "pus_cell_clumps":"pcc", "bacteria":"ba",
        "blood_glucose_random":"bgr", "blood_urea":"bu",
        "serum_creatinine":"sc", "sodium":"sod", "potassium":"pot",
        "haemoglobin":"hemo", "packed_cell_volume":"pcv",
        "white_blood_cell_count":"wbcc", "red_blood_cell_count":"rbcc",
        "hypertension":"htn", "diabetes_mellitus":"dm",
        "coronary_artery_disease":"cad", "appetite":"appet",
        "peda_edema":"pe", "aanemia":"ane",
    }
    # Check if model used short names by checking any short name is in expected
    short_names = set(DESC_TO_SHORT.values())
    if any(e in short_names for e in expected):
        X = X_full.rename(columns=DESC_TO_SHORT)
        print("[OK] Renamed to short column names to match model")
    else:
        X = X_full
        print("[OK] Using descriptive column names")
except Exception:
    X = X_full

print(f"[OK] Using {X.shape[1]} features")

# same 70/30 split with same random_state=42 as training
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"Test set: {len(X_test)} patients")

# Encode categorical columns — old pipeline has no encoder step inside
from sklearn.preprocessing import LabelEncoder as _LE
X_test = X_test.copy()
for col in X_test.columns:
    if X_test[col].dtype == object or str(X_test[col].dtype) == "string":
        X_test[col] = X_test[col].fillna("missing")
        _le = _LE()
        X_test[col] = _le.fit_transform(X_test[col].astype(str))
    else:
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
print("[OK] Categorical columns encoded")

# ── 3. Plot XGBoost ROC curve ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for name, pipe in pipelines.items():
    y_prob       = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc      = auc(fpr, tpr)

    # Main ROC curve
    ax.plot(fpr, tpr,
            color="#1D9E75",
            linewidth=2.5,
            label=f"{name}  (AUC = {roc_auc:.3f})")

    # Shade area under curve
    ax.fill_between(fpr, tpr, alpha=0.08, color="#1D9E75")

    # Mark the optimal threshold point (closest to top-left corner)
    opt_idx = np.argmax(tpr - fpr)
    ax.scatter(fpr[opt_idx], tpr[opt_idx],
               color="#1D9E75", s=90, zorder=5,
               label=f"Optimal threshold = {thresholds[opt_idx]:.2f}")

    print(f"\n{name} ROC AUC: {roc_auc:.4f}")
    print(f"Optimal threshold: {thresholds[opt_idx]:.3f}")
    print(f"  TPR at optimal: {tpr[opt_idx]:.3f}")
    print(f"  FPR at optimal: {fpr[opt_idx]:.3f}")

# Random classifier baseline
ax.plot([0, 1], [0, 1], ":",
        color="#999999", linewidth=1.2,
        label="Random classifier  (AUC = 0.500)")

# Formatting
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
ax.set_xlabel("False Positive Rate  (1 - Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate  (Sensitivity / Recall)", fontsize=12)
ax.set_title("ROC Curve — XGBoost\nCKD Risk Assessment",
             fontsize=14, fontweight="bold", color="#1a1a2e", pad=12)
ax.legend(loc="lower right", fontsize=10,
          framealpha=0.92, edgecolor="#cccccc")
ax.grid(True, alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out = "xgboost_roc_curve.jpg"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved: {out}")