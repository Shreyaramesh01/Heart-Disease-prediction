# Heart Disease Prediction System (Flask + ML)

## Run
```
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000  -> login `admin` / `admin123`

## Dataset
Put the UCI/Kaggle `heart.csv` in this folder (columns: age, sex, cp, trestbps, chol, fbs,
restecg, thalach, exang, oldpeak, slope, ca, thal, target). Delete `model.pkl` or open `/retrain`
after adding it. Without it the app trains on synthetic demo data.

## MySQL (optional, SQLite is default)
Create DB: `CREATE DATABASE heart_db;` then
Windows: `set DATABASE_URL=mysql+pymysql://root:password@localhost/heart_db`
Linux/Mac: `export DATABASE_URL=...`
