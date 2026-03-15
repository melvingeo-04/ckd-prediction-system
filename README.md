# KidneyGuard — CKD Risk Assessment System

AI-powered Chronic Kidney Disease screening platform built with Flask + RandomForest + SHAP.

## Setup Instructions

### 1. Place your trained model
Copy your trained model file into the app folder:
```
ckd_best_model.pkl  →  ckd_app/ckd_best_model.pkl
```

### 2. Install dependencies
```bash
cd ckd_app
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

---

## Features
- Home page with kidney health education
- Patient registration with auto-generated Patient ID (PID0001, PID0002...)
- Patient search by ID or name
- CKD risk assessment form (manual entry or CSV/Excel upload)
- AI prediction with risk score (0–100%) and risk level
- SHAP explainability chart showing feature contributions
- Full test report history per patient
- Print report functionality

## Project Structure
```
ckd_app/
├── app.py              # Flask routes & prediction logic
├── database.py         # SQLite database operations
├── ckd_best_model.pkl  # Your trained RandomForest model (place here)
├── ckd.db              # SQLite DB (auto-created on first run)
├── requirements.txt
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── register.html
│   ├── profile.html
│   ├── predict.html
│   ├── result.html
│   ├── patients.html
│   ├── search.html
│   ├── edu_diseases.html
│   ├── edu_risk_factors.html
│   ├── edu_diet.html
│   └── edu_prevention.html
└── static/
    ├── css/style.css
    └── js/main.js
```

## Input Features Used for Prediction
| Feature | Description |
|---------|-------------|
| age | Age in years |
| bp | Blood Pressure (mm/Hg) |
| sc | Serum Creatinine (mgs/dl) |
| bu | Blood Urea (mgs/dl) |
| hemo | Hemoglobin (gms) |
| bgr | Blood Glucose Random (mgs/dl) |
| al | Albumin (0-5) |
| sg | Specific Gravity |
| htn | Hypertension (1=Yes, 0=No) |
| dm | Diabetes Mellitus (1=Yes, 0=No) |
