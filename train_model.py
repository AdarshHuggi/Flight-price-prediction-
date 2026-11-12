"""Train a reproducible flight fare estimator from the cleaned CSV."""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "EasyMyTrip_yatra_Cleaned.csv"
MODEL_PATH = ROOT / "flight_price_model.joblib"
METRICS_PATH = ROOT / "model_metrics.json"

CATEGORICAL = ["Airline", "Source", "Destination", "Month", "Route", "stop"]
NUMERIC = ["Day", "Dep_hour", "Dep_min", "Dur_hour", "Dur_min", "Arr_hour", "Arr_min"]
FEATURES = CATEGORICAL + NUMERIC


def load_data(path=DATA_PATH):
    data = pd.read_csv(path)
    # The source file includes an unnamed row-index column; it is not a model feature.
    data = data.loc[:, ~data.columns.astype(str).str.startswith("Unnamed")]
    data["price"] = pd.to_numeric(data["price"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    for column in NUMERIC:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["price"])
    return data


def train(data=None):
    data = load_data() if data is None else data
    X, y = data[FEATURES], data["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocess = ColumnTransformer([
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                   ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
        ("numeric", SimpleImputer(strategy="median"), NUMERIC),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2,
                                         random_state=42, n_jobs=-1)),
    ])
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    metrics = {
        "rows": int(len(data)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "mae": round(float(mean_absolute_error(y_test, predicted)), 2),
        "rmse": round(float(mean_squared_error(y_test, predicted) ** 0.5), 2),
        "r2": round(float(r2_score(y_test, predicted)), 4),
        "data_file": DATA_PATH.name,
        "note": "Random holdout metrics on a 2021 fare snapshot; not a live quote or guarantee.",
    }
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
