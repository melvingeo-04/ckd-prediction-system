from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import pickle, shap, json, os, io, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import wraps
from database import (init_db, add_doctor, get_doctor_by_username,
                      add_patient, get_patient, search_patients, add_report,
                      get_reports, get_all_patients, delete_patient)

app = Flask(__name__)
app.secret_key = "ckd-risk-assessment-2024"

# ── Load model + encoder + metadata ─────────────────────────
model        = None
ord_enc      = None
cat_cols     = []
model_meta   = {}

if os.path.exists("ckd_best_model.pkl"):
    with open("ckd_best_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("[OK] Model loaded.")

if os.path.exists("ckd_ord_enc.pkl"):
    with open("ckd_ord_enc.pkl", "rb") as f:
        enc_data  = pickle.load(f)
    ord_enc   = enc_data["encoder"]
    cat_cols  = enc_data["categorical_cols"]
    print(f"[OK] Encoder loaded. Categorical cols: {cat_cols}")

if os.path.exists("model_meta.json"):
    with open("model_meta.json") as f:
        model_meta = json.load(f)
    print(f"[OK] Meta loaded. Best model: {model_meta.get('best_model')}")

# ── 10 core input features — easy to collect clinically ──────
FEATURES = ["age","bp","sc","bu","hemo","bgr","al","sg","htn","dm"]

FEATURE_LABELS = {
    "age":  "Age (years)",
    "bp":   "Blood Pressure (mm/Hg)",
    "sc":   "Serum Creatinine (mgs/dl)",
    "bu":   "Blood Urea (mgs/dl)",
    "hemo": "Hemoglobin (gms)",
    "bgr":  "Blood Glucose Random (mgs/dl)",
    "al":   "Albumin",
    "sg":   "Specific Gravity",
    "htn":  "Hypertension",
    "dm":   "Diabetes Mellitus",
}

# All 24 model features — fallback if model file not loaded yet
ALL_FEATURES = ["age","bp","sg","al","su","rbc","pc","pcc","ba","bgr",
                "bu","sc","sod","pot","hemo","pcv","wc","rc",
                "htn","dm","cad","appet","pe","ane"]

# Defaults for the 14 features not collected from the form
# These are typical healthy values — model trained on the same UCI population
DEFAULTS = {"su":0,"rbc":1,"pc":1,"pcc":0,"ba":0,
            "sod":135,"pot":4.5,"pcv":44,"wc":7800,"rc":5.2,
            "wbcc":7800,"rbcc":5.2,
            "cad":0,"appet":1,"pe":0,"ane":0}

# Override ALL_FEATURES from saved metadata if available
if model_meta.get("all_features"):
    ALL_FEATURES = model_meta["all_features"]
    print(f"[OK] All features ({len(ALL_FEATURES)}) loaded from meta.")

init_db()

# ── Password helpers ─────────────────────────────────────────
try:
    import bcrypt
    def hash_password(pw):
        return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    def check_password(pw, hashed):
        return bcrypt.checkpw(pw.encode(), hashed.encode())
except ImportError:
    import hashlib
    def hash_password(pw):
        return hashlib.sha256(pw.encode()).hexdigest()
    def check_password(pw, hashed):
        return hashlib.sha256(pw.encode()).hexdigest() == hashed

# ── Auth decorator ───────────────────────────────────────────
def doctor_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "doctor_id" not in session:
            flash("Doctor login required.", "warning")
            return redirect(url_for("doctor_login"))
        return f(*args, **kwargs)
    return decorated

def current_doctor():
    if "doctor_id" in session:
        from database import get_doctor_by_id
        return get_doctor_by_id(session["doctor_id"])
    return None

# ── Helper: safely convert db row to plain dict ──────────────
def row_to_dict(row):
    """Convert sqlite3.Row or dict to a JSON-serialisable plain dict."""
    if row is None:
        return None
    if isinstance(row, dict):
        return row
    return dict(row)

# ── Helpers ──────────────────────────────────────────────────
def build_full_row(inp):
    """Build a DataFrame row with all 24 model features and
    apply the same OrdinalEncoder used during training.
    """
    row = dict(DEFAULTS)
    # Numeric inputs from the form
    for k in FEATURES:
        if k in inp and inp[k] not in (None, ""):
            row[k] = float(inp[k])
    df_row = pd.DataFrame([[row.get(f, 0) for f in ALL_FEATURES]],
                          columns=ALL_FEATURES)
    # Apply ordinal encoding to categorical columns if encoder available
    if ord_enc is not None and cat_cols:
        present = [c for c in cat_cols if c in df_row.columns]
        if present:
            df_row[present] = df_row[present].astype(str)
            df_row[present] = ord_enc.transform(df_row[present])
    return df_row

def shap_chart(shap_vals, feature_vals):
    labels = [FEATURE_LABELS.get(f, f) for f in FEATURES]
    colors = ["#e74c3c" if v > 0 else "#27ae60" for v in shap_vals]
    order  = np.argsort(np.abs(shap_vals))[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh([labels[i] for i in order[::-1]],
                   [shap_vals[i] for i in order[::-1]],
                   color=[colors[i] for i in order[::-1]], height=0.55)
    ax.axvline(0, color="#555", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on CKD risk)", fontsize=10)
    ax.set_title("Feature Contribution to Prediction", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)
    for bar, v in zip(bars, [shap_vals[i] for i in order[::-1]]):
        ax.text(v + (0.002 if v >= 0 else -0.002), bar.get_y() + bar.get_height()/2,
                f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()

# ════════════════════════════════════════════════════════════
# PUBLIC ROUTES
# ════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("home.html", doctor=current_doctor(), public_page=True)

@app.route("/education/<page>")
def education(page):
    valid = ["diseases","risk-factors","diet","prevention"]
    if page not in valid:
        return redirect(url_for("home"))
    return render_template(f"edu_{page.replace('-','_')}.html", doctor=current_doctor(), public_page=True)

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        data = request.form.to_dict()
        doc  = current_doctor()
        data["doctor_id"] = doc["did"] if doc else None
        pid  = add_patient(data)
        return redirect(url_for("public_predict", pid=pid))
    return render_template("register.html", doctor=current_doctor(), public_page=True)

# ── /api/lookup — JSON endpoint for inline PID search ────────
@app.route("/api/lookup")
def api_lookup():
    pid = request.args.get("pid", "").strip().upper()
    if not pid:
        return jsonify({"found": False})

    patient = get_patient(pid)
    if not patient:
        return jsonify({"found": False})

    # Convert to plain dict so jsonify works regardless of db row type
    patient_dict = row_to_dict(patient)

    reports = get_reports(pid)
    safe_reports = []
    for r in reports:
        rd = row_to_dict(r)
        rd.pop("chart_b64", None)   # strip large base64 blob
        # Ensure score is a plain Python type
        if "score" in rd:
            rd["score"] = float(rd["score"]) if rd["score"] is not None else None
        safe_reports.append(rd)

    return jsonify({
        "found":   True,
        "patient": patient_dict,
        "reports": safe_reports
    })

@app.route("/predict/<pid>", methods=["GET","POST"])
def public_predict(pid):
    patient = get_patient(pid)
    if not patient:
        flash("Patient not found.", "danger")
        return redirect(url_for("register"))

    if request.method == "POST":
        inp   = request.form.to_dict()
        df    = build_full_row(inp)
        prob  = float(model.predict_proba(df)[0][1])
        score = round(prob * 100, 1)
        if score < 30:   level, level_class = "Low Risk",      "low"
        elif score < 60: level, level_class = "Moderate Risk", "moderate"
        else:            level, level_class = "High Risk",     "high"

        try:
            inner   = model.named_steps["model"]
            imp     = model.named_steps["imputer"]
            scl     = model.named_steps["scaler"]
            df_proc = scl.transform(imp.transform(df))
            explainer = shap.TreeExplainer(inner)
            shap_all  = explainer.shap_values(df_proc, check_additivity=False)
            if isinstance(shap_all, list): sv = shap_all[1][0]
            elif shap_all.ndim == 3:       sv = shap_all[0, :, 1]
            else:                          sv = shap_all[0]
            feat_idx  = [ALL_FEATURES.index(f) for f in FEATURES]
            shap_vals = sv[feat_idx]
            chart_b64 = shap_chart(shap_vals, df[FEATURES].values[0])
            order     = np.argsort(np.abs(shap_vals))[::-1]
            explanations = [
                f"{FEATURE_LABELS[FEATURES[i]]} {'increased' if shap_vals[i]>0 else 'lowered'} the predicted risk."
                for i in order[:5]
            ]
        except Exception as e:
            chart_b64    = None
            explanations = [f"SHAP unavailable: {e}"]
            shap_vals    = [0.0] * len(FEATURES)

        add_report({
            "pid":         pid,
            "inputs":      json.dumps({k: inp.get(k) for k in FEATURES}),
            "score":       score,
            "level":       level,
            "shap_values": json.dumps([round(float(v), 4) for v in shap_vals]),
            "chart_b64":   chart_b64 or ""
        })
        return render_template("result.html",
            patient=patient, score=score, level=level, level_class=level_class,
            chart_b64=chart_b64, explanations=explanations,
            shap_vals=[round(float(v), 4) for v in shap_vals],
            shap_labels=[FEATURE_LABELS[f] for f in FEATURES],
            inputs={k: inp.get(k) for k in FEATURES},
            feature_labels=FEATURE_LABELS, doctor=current_doctor(), public_page=True)

    return render_template("predict.html", patient=patient,
                           feature_labels=FEATURE_LABELS, features=FEATURES,
                           doctor=current_doctor(), public_page=True)

# ── View single report ────────────────────────────────────────
@app.route("/result/<int:report_id>")
def public_result_by_id(report_id):
    from database import get_conn
    conn = get_conn()
    row  = conn.execute("SELECT * FROM reports WHERE report_id=?", (report_id,)).fetchone()
    conn.close()
    if not row:
        flash("Report not found.", "danger")
        return redirect(url_for("register"))
    r       = dict(row)
    patient = get_patient(r["pid"])
    inp     = json.loads(r["inputs"] or "{}")
    score   = r["score"]
    level   = r["level"]
    if score < 30:   level_class = "low"
    elif score < 60: level_class = "moderate"
    else:            level_class = "high"
    try:
        shap_vals = json.loads(r["shap_values"] or "[]")
        order     = list(np.argsort(np.abs(shap_vals))[::-1])
        explanations = [
            f"{FEATURE_LABELS.get(FEATURES[i], FEATURES[i])} {'increased' if shap_vals[i]>0 else 'lowered'} the predicted risk."
            for i in order[:5] if i < len(FEATURES)
        ]
    except Exception:
        explanations = []
        shap_vals    = []
    return render_template("result.html",
        patient=patient, score=score, level=level, level_class=level_class,
        chart_b64=r.get("chart_b64") or None,
        explanations=explanations,
        shap_vals=[round(float(v), 4) for v in shap_vals] if shap_vals else [],
        shap_labels=[FEATURE_LABELS[f] for f in FEATURES],
        inputs=inp, feature_labels=FEATURE_LABELS, doctor=current_doctor(), public_page=True)

# ════════════════════════════════════════════════════════════
# DOCTOR ROUTES
# ════════════════════════════════════════════════════════════

@app.route("/doctor/login", methods=["GET","POST"])
def doctor_login():
    if "doctor_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username","").strip().lower()
        password = request.form.get("password","")
        doctor   = get_doctor_by_username(username)
        if doctor and check_password(password, doctor["password"]):
            session["doctor_id"]   = doctor["did"]
            session["doctor_name"] = doctor["name"]
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "danger")
    return render_template("doctor_login.html", show_register=False)

@app.route("/doctor/register", methods=["POST"])
def doctor_register():
    data = request.form.to_dict()
    if get_doctor_by_username(data.get("username","").lower()):
        flash("Username already taken. Choose another.", "danger")
        return render_template("doctor_login.html", show_register=True)
    from database import get_doctor_by_email
    if get_doctor_by_email(data.get("email","").lower()):
        flash("Email already registered.", "danger")
        return render_template("doctor_login.html", show_register=True)
    data["password"] = hash_password(data["password"])
    add_doctor(data)
    flash("Account created! Please sign in.", "success")
    return render_template("doctor_login.html", show_register=False)

@app.route("/doctor/logout")
def doctor_logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/dashboard")
@doctor_required
def dashboard():
    doctor = current_doctor()
    all_p  = get_all_patients(doctor["did"])
    # Count total reports (predictions run) for this doctor's patients
    from database import get_conn
    conn = get_conn()
    pids = [p["pid"] for p in all_p]
    total_predictions = 0
    if pids:
        placeholders = ",".join("?" * len(pids))
        total_predictions = conn.execute(
            f"SELECT COUNT(*) FROM reports WHERE pid IN ({placeholders})", pids
        ).fetchone()[0]
    conn.close()
    return render_template("dashboard.html", doctor=doctor,
                           patients=all_p, meta=model_meta,
                           total_predictions=total_predictions)

@app.route("/patients")
@doctor_required
def patients():
    doctor = current_doctor()
    all_p  = get_all_patients(doctor["did"])
    return render_template("patients.html", patients=all_p, doctor=doctor)

@app.route("/patient/<pid>")
def patient_profile(pid):
    patient = get_patient(pid)
    if not patient:
        flash("Patient not found.", "danger")
        return redirect(url_for("register"))
    reports = get_reports(pid)
    return render_template("profile.html", patient=patient,
                           reports=reports, doctor=current_doctor())

@app.route("/search")
@doctor_required
def search():
    q       = request.args.get("q","").strip()
    doctor  = current_doctor()
    results = search_patients(q, doctor["did"]) if q else []
    return render_template("search.html", results=results,
                           query=q, doctor=doctor)

@app.route("/delete_patient/<pid>", methods=["POST"])
@doctor_required
def delete_patient_route(pid):
    doctor = current_doctor()
    delete_patient(pid, doctor["did"])
    return redirect(url_for("patients"))

@app.route("/model-info")
def model_info():
    return render_template("model_info.html",
                           doctor=current_doctor(),
                           public_page=True,
                           meta=model_meta)

if __name__ == "__main__":
    app.run(debug=True, port=5000)