"""Train the modular ensemble on the prepared real-world NHANES 2017-2018 data.

Run:
    python train_nhanes_model.py

This is intentionally separate from train_model.py, which uses the synthetic
dataset and a richer set of synthetic-only input fields.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = "nhanes_2017_2018/nhanes_diabetes_analysis.csv"
OUTPUT = "nhanes_2017_2018/nhanes_ensemble.joblib"
TARGET = "diagnosed_diabetes"

# These are the closest real NHANES 2017-18 fields to the research subdomains.
# NHANES lacks a direct family-history-of-diabetes and screen-time variable.
DOMAINS = {
    "Body / Weight": ["age", "bmi", "waist_to_hip_ratio"],
    "Heart": ["hypertension_history", "cardiovascular_history", "systolic_bp", "diastolic_bp", "heart_rate"],
    "Cholesterol": ["cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides"],
    "Blood Sugar": ["glucose_fasting", "insulin_level", "hba1c"],
    "Family History": [],  # no direct matching variable in selected NHANES cycle
    "Exercise": ["physical_activity_minutes_per_week"],
    "Diet / Habits": ["diet_energy_kcal", "diet_sugar_g", "diet_fiber_g", "diet_saturated_fat_g", "smoking_status", "alcohol_drinks_per_day"],
    "Sleep": ["sleep_hours_per_day"],
    "Income / Education": ["income_code", "education_code", "gender_code", "ethnicity_code", "poverty_income_ratio"],
}


def make_model(features: list[str], frame: pd.DataFrame) -> Pipeline:
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in features if c not in numeric]
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent", add_indicator=True)), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocess", preprocessing), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def prevalence_threshold(model, y_train, x_evaluation) -> float:
    probability = np.sort(model.predict_proba(x_evaluation)[:, 1])[::-1]
    positive_count = round(float(y_train.mean()) * len(probability))
    if positive_count <= 0:
        return 1.0
    if positive_count >= len(probability):
        return 0.0
    return float((probability[positive_count - 1] + probability[positive_count]) / 2)


def metrics(model, x, y, threshold) -> dict[str, float]:
    probability = model.predict_proba(x)[:, 1]
    prediction = (probability >= threshold).astype(int)
    return {
        "accuracy": round(accuracy_score(y, prediction), 3),
        "f1": round(f1_score(y, prediction), 3),
        "roc_auc": round(roc_auc_score(y, probability), 3),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y, prediction).tolist(),
    }


def main():
    data = pd.read_csv(DATA)
    data = data[(data["age"] >= 18) & data[TARGET].isin([0, 1])].copy()
    # Avoid learning from the survey weights or respondent ID.
    x = data.drop(columns=[TARGET])
    y = data[TARGET].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)

    domain_models, train_scores, test_scores, unavailable = {}, {}, {}, []
    for domain, features in DOMAINS.items():
        if not features:
            unavailable.append(domain)
            continue
        model = make_model(features, x_train)
        model.fit(x_train[features], y_train)
        domain_models[domain] = model
        train_scores[domain] = model.predict_proba(x_train[features])[:, 1]
        test_scores[domain] = model.predict_proba(x_test[features])[:, 1]

    meta_train = pd.DataFrame(train_scores, index=x_train.index)
    meta_test = pd.DataFrame(test_scores, index=x_test.index)
    orchestrator = LogisticRegression(max_iter=2000, class_weight="balanced")
    orchestrator.fit(meta_train, y_train)
    threshold = prevalence_threshold(orchestrator, y_train, meta_test)

    feature_list = [feature for features in DOMAINS.values() for feature in features]
    numeric_features = [f for f in feature_list if pd.api.types.is_numeric_dtype(x_train[f])]
    categorical_features = [f for f in feature_list if f not in numeric_features]
    bundle = {
        "dataset": "NHANES 2017-2018 (real-world, public CDC survey)",
        "target_definition": "Self-reported doctor-diagnosed diabetes (DIQ010)",
        "domain_models": domain_models,
        "orchestrator": orchestrator,
        "domains": {key: value for key, value in DOMAINS.items() if value},
        "unavailable_domains": unavailable,
        "metrics": metrics(orchestrator, meta_test, y_test, threshold),
        "sample_sizes": {"training": len(x_train), "test": len(x_test), "total": len(data)},
        "positive_rate": round(float(y.mean()), 3),
        "background_scores": meta_train.sample(min(100, len(meta_train)), random_state=42),
        "default_values": x_train[numeric_features].median().to_dict(),
        "category_values": {f: sorted(x_train[f].dropna().astype(str).unique().tolist()) for f in categorical_features},
        "notes": [
            "NHANES data require survey weights, strata and PSU variables for population-representative estimates.",
            "Family History is omitted because no direct family-history-of-diabetes variable was included in this selected cycle.",
            "The output validates model behavior on one cross-sectional U.S. survey cycle; it is not a prospective clinical-risk model.",
        ],
    }
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUTPUT)
    print("Saved model:", Path(OUTPUT).resolve())
    print("Sample sizes:", json.dumps(bundle["sample_sizes"]))
    print("Positive class rate:", bundle["positive_rate"])
    print("Held-out test metrics:", json.dumps(bundle["metrics"], indent=2))
    print("Unavailable subdomains:", ", ".join(unavailable) or "None")


if __name__ == "__main__":
    main()
