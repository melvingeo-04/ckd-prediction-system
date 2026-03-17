import sqlite3
from datetime import datetime

DB = "ckd.db"

def get_conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS doctors (
        did         TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        specialization TEXT,
        hospital    TEXT,
        email       TEXT UNIQUE NOT NULL,
        username    TEXT UNIQUE NOT NULL,
        password    TEXT NOT NULL,
        reg_date    TEXT
    );
    CREATE TABLE IF NOT EXISTS patients (
        pid         TEXT PRIMARY KEY,
        doctor_id   TEXT,
        name        TEXT NOT NULL,
        age         INTEGER,
        gender      TEXT,
        reg_date    TEXT,
        FOREIGN KEY(doctor_id) REFERENCES doctors(did)
    );
    CREATE TABLE IF NOT EXISTS reports (
        report_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        pid         TEXT,
        test_date   TEXT,
        inputs      TEXT,
        score       REAL,
        level       TEXT,
        shap_values TEXT,
        chart_b64   TEXT,
        FOREIGN KEY(pid) REFERENCES patients(pid)
    );
    """)
    conn.commit()
    conn.close()

# ── Doctor functions ─────────────────────────────────────────
def _next_did():
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) as c FROM doctors").fetchone()["c"] + 1
    conn.close()
    return f"DID{n:04d}"

def add_doctor(data):
    did  = _next_did()
    conn = get_conn()
    conn.execute("""
        INSERT INTO doctors (did,name,specialization,hospital,email,username,password,reg_date)
        VALUES (?,?,?,?,?,?,?,?)
    """, (did, data.get("name"), data.get("specialization"),
          data.get("hospital"), data.get("email","").lower(),
          data.get("username","").lower(),
          data.get("password"), datetime.now().strftime("%Y-%m-%d")))
    conn.commit()
    conn.close()
    return did

def get_doctor_by_email(email):
    conn = get_conn()
    row  = conn.execute("SELECT * FROM doctors WHERE email=?", (email.lower(),)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_doctor_by_username(username):
    conn = get_conn()
    row  = conn.execute("SELECT * FROM doctors WHERE username=?", (username.lower(),)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_doctor_by_id(did):
    conn = get_conn()
    row  = conn.execute("SELECT * FROM doctors WHERE did=?", (did,)).fetchone()
    conn.close()
    return dict(row) if row else None

# ── Patient functions ────────────────────────────────────────
def _next_pid():
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) as c FROM patients").fetchone()["c"] + 1
    conn.close()
    return f"PID{n:04d}"

def add_patient(data):
    pid  = _next_pid()
    conn = get_conn()
    conn.execute("""
        INSERT INTO patients (pid,doctor_id,name,age,gender,reg_date)
        VALUES (?,?,?,?,?,?)
    """, (pid, data.get("doctor_id"), data.get("name"),
          data.get("age"), data.get("gender"),
          datetime.now().strftime("%Y-%m-%d")))
    conn.commit()
    conn.close()
    return pid

def get_patient(pid, doctor_id=None):
    conn = get_conn()
    if doctor_id:
        row = conn.execute(
            "SELECT * FROM patients WHERE pid=? AND doctor_id=?", (pid, doctor_id)
        ).fetchone()
    else:
        row = conn.execute("SELECT * FROM patients WHERE pid=?", (pid,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_patients(doctor_id=None):
    # Doctors can see ALL patients — including public patients registered without login
    conn = get_conn()
    rows = conn.execute("SELECT * FROM patients ORDER BY rowid DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def search_patients(q, doctor_id=None):
    conn = get_conn()
    q    = f"%{q}%"
    if doctor_id:
        rows = conn.execute(
            "SELECT * FROM patients WHERE doctor_id=? AND (pid LIKE ? OR name LIKE ?)",
            (doctor_id, q, q)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM patients WHERE pid LIKE ? OR name LIKE ?", (q, q)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_patient(pid, doctor_id=None):
    conn = get_conn()
    if doctor_id:
        conn.execute("DELETE FROM reports WHERE pid=?", (pid,))
        conn.execute("DELETE FROM patients WHERE pid=? AND doctor_id=?", (pid, doctor_id))
    else:
        conn.execute("DELETE FROM reports WHERE pid=?", (pid,))
        conn.execute("DELETE FROM patients WHERE pid=?", (pid,))
    conn.commit()
    conn.close()

# ── Report functions ─────────────────────────────────────────
def add_report(data):
    conn = get_conn()
    conn.execute("""
        INSERT INTO reports (pid,test_date,inputs,score,level,shap_values,chart_b64)
        VALUES (?,?,?,?,?,?,?)
    """, (data["pid"], datetime.now().strftime("%Y-%m-%d %H:%M"),
          data["inputs"], data["score"], data["level"],
          data["shap_values"], data["chart_b64"]))
    conn.commit()
    conn.close()

def get_reports(pid):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM reports WHERE pid=? ORDER BY report_id DESC", (pid,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]