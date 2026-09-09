"""Train the subdomain-aware diabetes-risk ensemble.

Run:
    py -3 train_model.py --data "path-to-Diabetes_and_LifeStyle_Dataset.csv"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "diagnosed_diabetes"

# diabetes_risk_score and diabetes_stage are deliberately excluded: they are
# outcome-like variables and would leak information about the diagnosis.
DOMAINS = {
    "Body / Weight": ["Age", "bmi", "waist_to_hip_ratio"],
    "Heart": ["hypertension_history", "cardiovascular_history", "systolic_bp", "diastolic_bp", "heart_rate"],
    "Cholesterol": ["cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides"],
    "Blood Sugar": ["glucose_fasting", "glucose_postprandial", "insulin_level", "hba1c"],
    "Family History": ["family_history_diabetes"],
    "Exercise": ["physical_activity_minutes_per_week"],
    "Diet / Habits": ["diet_score", "smoking_status", "alcohol_consumption_per_week", "screen_time_hours_per_day"],
    "Sleep": ["sleep_hours_per_day"],
    "Income / Education": ["income_level", "education_level", "employment_status", "gender", "ethnicity"],
}


def make_model(features: list[str], frame: pd.DataFrame) -> Pipeline:
    """A compact preprocessing + interpretable classifier pipeline."""
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in features if c not in numeric]
    preprocess = ColumnTransformer([
        ("number", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("category", Pipeline([("fill", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("prepare", preprocess), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])


def score(model, x_test, y_test) -> dict[str, float]:
    probability = model.predict_proba(x_test)[:, 1]
    predicted = (probability >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y_test, predicted), 3),
        "f1": round(f1_score(y_test, predicted), 3),
        "roc_auc": round(roc_auc_score(y_test, probability), 3),
    }


def train(data_path: str, output_path: str = "diabetes_ensemble.joblib") -> dict:
    data = pd.read_csv(data_path)
    needed = sorted({column for columns in DOMAINS.values() for column in columns} | {TARGET})
    missing = set(needed) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    x = data.drop(columns=[TARGET])
    y = data[TARGET].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )

    domain_models, train_scores, test_scores = {}, {}, {}
    for domain, features in DOMAINS.items():
        model = make_model(features, x_train)
        model.fit(x_train[features], y_train)
        domain_models[domain] = model
        train_scores[domain] = model.predict_proba(x_train[features])[:, 1]
        test_scores[domain] = model.predict_proba(x_test[features])[:, 1]

    domain_names = list(DOMAINS)
    meta_train = pd.DataFrame(train_scores, columns=domain_names, index=x_train.index)
    meta_test = pd.DataFrame(test_scores, columns=domain_names, index=x_test.index)
    orchestrator = LogisticRegression(max_iter=1000, class_weight="balanced")
    orchestrator.fit(meta_train, y_train)

    bundle = {
        "domain_models": domain_models,
        "orchestrator": orchestrator,
        "domains": DOMAINS,
        "default_values": x_train[list({c for v in DOMAINS.values() for c in v})].median(numeric_only=True).to_dict(),
        "category_values": {c: sorted(x_train[c].dropna().astype(str).unique().tolist()) for c in x_train.select_dtypes(include="object")},
        "metrics": score(orchestrator, meta_test, y_test),
        "background_scores": meta_train.sample(min(100, len(meta_train)), random_state=42),
        "target": TARGET,
    }
    joblib.dump(bundle, output_path)
    print("Saved model:", Path(output_path).resolve())
    print("Test metrics:", json.dumps(bundle["metrics"], indent=2))
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the CSV dataset")
    parser.add_argument("--output", default="diabetes_ensemble.joblib")
    args = parser.parse_args()
    train(args.data, args.output)
