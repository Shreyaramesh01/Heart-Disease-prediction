import os, io, base64
from datetime import datetime
from functools import wraps
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from ml import load_or_train, FEATURES, BASE

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me")
# SQLite by default. For MySQL set e.g.:
# DATABASE_URL=mysql+pymysql://root:password@localhost/heart_db
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///" + os.path.join(BASE, "heart.db"))
db = SQLAlchemy(app)

ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "admin123")


class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), index=True)
    name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    details = db.Column(db.Text)          # all 13 inputs as text
    result = db.Column(db.String(30))
    probability = db.Column(db.Float)
    algorithm = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.now)


# (name, label, kind, options-or-range)
FIELDS = [
    ("patient_id", "Patient ID", "text", None),
    ("patient_name", "Patient Name", "text", None),
    ("age", "Age", "number", (1, 120)),
    ("sex", "Gender", "select", [("1", "Male"), ("0", "Female")]),
    ("cp", "Chest Pain", "select", [("0", "Typical"), ("1", "Atypical"), ("2", "Non-anginal"), ("3", "Asymptomatic")]),
    ("trestbps", "Resting BP", "number", (80, 200)),
    ("chol", "Cholesterol", "number", (100, 400)),
    ("fbs", "Fasting Blood Sugar > 120", "select", [("0", "False"), ("1", "True")]),
    ("restecg", "Rest ECG", "select", [("0", "Normal"), ("1", "ST-T abnormality"), ("2", "LV hypertrophy")]),
    ("thalach", "Max Heart Rate", "number", (60, 220)),
    ("exang", "Exercise Angina", "select", [("0", "No"), ("1", "Yes")]),
    ("oldpeak", "Oldpeak", "number", (0, 6)),
    ("slope", "Slope", "select", [("0", "Upsloping"), ("1", "Flat"), ("2", "Downsloping")]),
    ("ca", "Major Vessels", "select", [("0", "0"), ("1", "1"), ("2", "2"), ("3", "3")]),
    ("thal", "Thal", "select", [("0", "Normal"), ("1", "Fixed defect"), ("2", "Reversible defect"), ("3", "Other")]),
]

bundle = load_or_train()


def login_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if not session.get("admin"):
            return redirect(url_for("login"))
        return f(*a, **kw)
    return wrapper


def make_graphs(metrics):
    names = [m["algorithm"] for m in metrics]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, key in zip(axes.flat, ["accuracy", "precision", "recall", "f1"]):
        ax.bar(names, [m[key] for m in metrics], color="#1f77b4")
        ax.set_title(key.capitalize())
        ax.tick_params(axis="x", rotation=40, labelsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == ADMIN_USER and request.form["password"] == ADMIN_PASS:
            session["admin"] = True
            return redirect(url_for("home"))
        flash("Invalid credentials")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def home():
    return render_template("index.html", fields=FIELDS, values={}, synthetic=bundle["synthetic"])


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    form = request.form
    # server-side validation
    try:
        for name, label, kind, opt in FIELDS:
            v = form.get(name, "").strip()
            if not v:
                raise ValueError(f"{label} is required")
            if kind == "number":
                x = float(v)
                if not opt[0] <= x <= opt[1]:
                    raise ValueError(f"{label} must be between {opt[0]} and {opt[1]}")
    except ValueError as e:
        flash(str(e))
        return render_template("index.html", fields=FIELDS, values=form, synthetic=bundle["synthetic"])

    row = pd.DataFrame([{f: float(form[f]) for f in FEATURES}], columns=FEATURES)
    best = bundle["best"]
    model = bundle["models"][best]
    pred = int(model.predict(row)[0])
    prob = round(float(model.predict_proba(row)[0][1]) * 100, 1)
    label = "Heart Disease" if pred == 1 else "No Heart Disease"

    db.session.add(Patient(
        patient_id=form["patient_id"], name=form["patient_name"], age=int(float(form["age"])),
        gender="Male" if form["sex"] == "1" else "Female",
        details=str({f: form[f] for f in FEATURES}), result=label, probability=prob, algorithm=best))
    db.session.commit()

    return render_template("result.html", label=label, pred=pred, prob=prob, best=best,
                           metrics=bundle["metrics"], graph=make_graphs(bundle["metrics"]))


@app.route("/history")
@login_required
def history():
    q = request.args.get("pid", "").strip()
    query = Patient.query.order_by(Patient.created_at.desc())
    if q:
        query = query.filter(Patient.patient_id == q)
    return render_template("history.html", rows=query.all(), q=q)


@app.route("/retrain")
@login_required
def retrain():
    global bundle
    bundle = load_or_train(force=True)
    flash("Models retrained")
    return redirect(url_for("home"))


with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)
