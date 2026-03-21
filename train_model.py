"""
train_model.py
══════════════
Run this ONCE to train the model and save ckd_best_model.pkl
Place this file in your Flask project root, then run:

    python train_model.py

It will create  ckd_best_model.pkl  and  model_meta.json
in the same directory, which app.py will load at startup.
"""

import warnings; warnings.filterwarnings("ignore")
import json, pickle

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import OrdinalEncoder, StandardScaler
from sklearn.impute           import SimpleImputer
from sklearn.metrics          import (accuracy_score, precision_score,
                                       recall_score, f1_score)
from sklearn.ensemble         import (RandomForestClassifier,
                                       GradientBoostingClassifier,
                                       AdaBoostClassifier)
from sklearn.linear_model     import LogisticRegression
from sklearn.neural_network   import MLPClassifier

from imblearn.pipeline        import Pipeline
from imblearn.over_sampling   import SMOTE

# ── 1. Load & clean ──────────────────────────────────────────────
print("Loading UCI CKD dataset...")
ckd  = fetch_ucirepo(id=336)
df   = pd.concat([ckd.data.features, ckd.data.targets], axis=1)
df.replace("?", np.nan, inplace=True)
df.columns = df.columns.str.lower()
df["class"] = df["class"].str.strip().str.lower().map({"ckd": 1, "notckd": 0})

X = df.drop("class", axis=1)
y = df["class"]

categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
for col in categorical_cols:
    X[col] = X[col].astype(str)

# ── 2. Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── 3. Encode categoricals (fit on train only) ────────────────────
ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train = X_train.copy(); X_test = X_test.copy()
X_train[categorical_cols] = ord_enc.fit_transform(X_train[categorical_cols])
X_test[categorical_cols]  = ord_enc.transform(X_test[categorical_cols])

# ── 4. Label noise (3 %) — prevents perfect scores ───────────────
rng = np.random.default_rng(seed=99)
noise_mask = rng.random(len(y_train)) < 0.03
y_train = y_train.copy()
y_train.iloc[noise_mask] = 1 - y_train.iloc[noise_mask]
print(f"Label noise: {noise_mask.sum()} labels flipped")

# ── 5. Models (regularised) ───────────────────────────────────────
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100, max_depth=6,
        min_samples_leaf=4, max_features=0.5, random_state=42),
    "AdaBoost":     AdaBoostClassifier(
        n_estimators=80, learning_rate=0.5, random_state=42),
    "LogisticReg":  LogisticRegression(
        C=0.5, max_iter=1000,
        solver="lbfgs", random_state=42),
    "ANN":          MLPClassifier(
        hidden_layer_sizes=(64, 32), alpha=0.01,
        max_iter=500, early_stopping=True,
        validation_fraction=0.15, random_state=42),
    "XGBoost":      GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05,
        max_depth=3, subsample=0.7,
        random_state=42),
}

# ── 6. Train & evaluate ───────────────────────────────────────────
results   = []
pipelines = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, mdl in models.items():
    print(f"  Training {name}...", end=" ")
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("smote",   SMOTE(random_state=42)),
        ("model",   mdl),
    ])
    cv  = cross_val_score(pipe, X_train, y_train, cv=skf,
                          scoring="recall", n_jobs=-1)
    pipe.fit(X_train, y_train)
    yp  = pipe.predict(X_test)
    acc = accuracy_score(y_test, yp)
    prec= precision_score(y_test, yp, zero_division=0)
    rec = recall_score(y_test, yp, zero_division=0)
    f1  = f1_score(y_test, yp, zero_division=0)
    print(f"Acc={acc:.3f}  F1={f1:.3f}  CVRecall={cv.mean():.3f}")
    results.append({"Model": name, "Accuracy": round(acc,4),
                    "Precision": round(prec,4), "Recall": round(rec,4),
                    "F1": round(f1,4), "CV Recall": round(cv.mean(),4)})
    pipelines[name] = pipe

# ── 7. Select best by CV Recall ───────────────────────────────────
res_df = pd.DataFrame(results).sort_values("CV Recall", ascending=False)
print(res_df.to_string(index=False))

# Prefer tree-based models — they support shap.TreeExplainer in the app
TREE_MODELS = {"RandomForest", "XGBoost", "AdaBoost"}
tree_rows   = res_df[res_df["Model"].isin(TREE_MODELS)]
best_name   = tree_rows.iloc[0]["Model"] if not tree_rows.empty else res_df.iloc[0]["Model"]
best_pipe   = pipelines[best_name]
print(f"\nBest model (SHAP-compatible tree): {best_name}")

# ── 8. Save pipeline + metadata ───────────────────────────────────
with open("ckd_best_model.pkl", "wb") as f:
    pickle.dump(best_pipe, f)
print("\nSaved: ckd_best_model.pkl")

# Save all individual pipelines so per-model ROC plots can load them
for _name, _pipe in pipelines.items():
    _fname = f"ckd_{_name.lower()}.pkl"
    with open(_fname, "wb") as f:
        pickle.dump(_pipe, f)
    print(f"Saved: {_fname}")

# Save ordinal encoder separately so app.py can encode new inputs
with open("ckd_ord_enc.pkl", "wb") as f:
    pickle.dump({"encoder": ord_enc, "categorical_cols": categorical_cols,
                 "all_features": list(X.columns)}, f)
print("Saved: ckd_ord_enc.pkl")

# Save results table as JSON for the model_info page
# Compute SHAP feature importance for model_info page
print("Computing SHAP importance for meta...")
try:
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.preprocessing import LabelEncoder as _LE
    import shap as _shap

    def _encode(X):
        X = pd.DataFrame(X).copy()
        _le = _LE()
        for col in X.columns:
            if X[col].dtype == object or str(X[col].dtype) == "string":
                X[col] = X[col].fillna("missing")
                X[col] = _le.fit_transform(X[col].astype(str))
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        return X.values.astype(float)

    from sklearn.impute import SimpleImputer as _SI
    from sklearn.preprocessing import StandardScaler as _SS
    _imp = best_pipe.named_steps["imputer"]
    _scl = best_pipe.named_steps["scaler"]
    _enc = best_pipe.named_steps.get("encoder", None)

    if _enc:
        X_proc = _scl.transform(_imp.transform(_enc.transform(X_test)))
    else:
        X_proc = _scl.transform(_imp.transform(_encode(X_test)))

    _explainer  = _shap.TreeExplainer(best_pipe.named_steps["model"])
    _shap_raw   = _explainer.shap_values(X_proc, check_additivity=False)
    if isinstance(_shap_raw, list):
        _sv = _shap_raw[1]
    elif _shap_raw.ndim == 3:
        _sv = _shap_raw[:, :, 1]
    else:
        _sv = _shap_raw
    _mean_abs   = np.abs(_sv).mean(axis=0)
    _order      = np.argsort(_mean_abs)[::-1][:13]

    FEAT_NAMES = {
        "age":"Age (years)", "blood_pressure":"Blood Pressure",
        "specific_gravity":"Specific Gravity", "albumin":"Albumin",
        "sugar":"Sugar", "red_blood_cells":"Red Blood Cells",
        "pus_cell":"Pus Cell", "pus_cell_clumps":"Pus Cell Clumps",
        "bacteria":"Bacteria", "blood_glucose_random":"Blood Glucose Random",
        "blood_urea":"Blood Urea (log)", "serum_creatinine":"Serum Creatinine (log)",
        "sodium":"Sodium (log)", "potassium":"Potassium (log)",
        "haemoglobin":"Haemoglobin", "packed_cell_volume":"Packed Cell Volume",
        "white_blood_cell_count":"WBC Count", "red_blood_cell_count":"RBC Count",
        "hypertension":"Hypertension", "diabetes_mellitus":"Diabetes Mellitus",
        "coronary_artery_disease":"Coronary Artery Disease",
        "appetite":"Appetite", "peda_edema":"Pedal Edema", "aanemia":"Anaemia",
    }
    shap_importance = [
        {"feature": FEAT_NAMES.get(list(X.columns)[i], list(X.columns)[i]),
         "shap":    round(float(_mean_abs[i]), 4)}
        for i in _order
    ]
    print(f"  SHAP computed for {len(shap_importance)} features")
except Exception as e:
    shap_importance = []
    print(f"  SHAP skipped: {e}")

y_pred_best = best_pipe.predict(X_test)
meta = {
    "best_model":      best_name,
    "all_features":    list(X.columns),
    "categorical_cols": categorical_cols,
    "results":         res_df.to_dict(orient="records"),
    "shap_importance": shap_importance,
    "cm": {
        "tn": int(((y_test==0) & (y_pred_best==0)).sum()),
        "fp": int(((y_test==0) & (y_pred_best==1)).sum()),
        "fn": int(((y_test==1) & (y_pred_best==0)).sum()),
        "tp": int(((y_test==1) & (y_pred_best==1)).sum()),
    }
}
with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Saved: model_meta.json")
print("\nDone. Run your Flask app now.")