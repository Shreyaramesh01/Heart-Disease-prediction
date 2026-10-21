"""Dataset loading, training of 7 algorithms, and best-model selection."""
import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
MODEL_FILE = os.path.join(BASE, "model.pkl")
CSV_FILE = os.path.join(BASE, "heart.csv")


def load_dataset():
    """Reads heart.csv (UCI/Kaggle Heart Disease). Falls back to synthetic demo data."""
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"thalch": "thalach", "output": "target", "num": "target"})
        df = df[FEATURES + ["target"]].apply(pd.to_numeric, errors="coerce").dropna()
        df["target"] = (df["target"] > 0).astype(int)
        return df, False
    # ---- synthetic demo data (only so the app runs without the CSV) ----
    rng = np.random.default_rng(42)
    n = 800
    df = pd.DataFrame({
        "age": rng.integers(29, 78, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(0, 4, n), "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 400, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n), "oldpeak": np.round(rng.uniform(0, 6, n), 1),
        "slope": rng.integers(0, 3, n), "ca": rng.integers(0, 4, n),
        "thal": rng.integers(0, 4, n)})
    z = (0.04 * (df.age - 54) + 0.6 * df.exang + 0.5 * df.oldpeak + 0.6 * df.ca
         - 0.03 * (df.thalach - 150) + 0.3 * (df.cp == 3) + rng.normal(0, 1, n))
    df["target"] = (z > 0.3).astype(int)
    return df, True


def train_all():
    df, synthetic = load_dataset()
    X, y = df[FEATURES], df["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1500, random_state=42),
    }
    fitted, metrics = {}, []
    for name, clf in models.items():
        pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)]).fit(Xtr, ytr)
        p = pipe.predict(Xte)
        metrics.append({
            "algorithm": name,
            "accuracy": round(accuracy_score(yte, p) * 100, 2),
            "precision": round(precision_score(yte, p, zero_division=0) * 100, 2),
            "recall": round(recall_score(yte, p, zero_division=0) * 100, 2),
            "f1": round(f1_score(yte, p, zero_division=0) * 100, 2)})
        fitted[name] = pipe
    best = max(metrics, key=lambda m: m["accuracy"])["algorithm"]
    bundle = {"models": fitted, "metrics": metrics, "best": best, "synthetic": synthetic}
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


def load_or_train(force=False):
    if not force and os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return train_all()
