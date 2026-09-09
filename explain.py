"""Prediction and SHAP-powered plain-language explanation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def predict_with_explanation(bundle: dict, person: dict) -> tuple[float, pd.DataFrame, str]:
    """Return final risk, actual SHAP domain contributions, and agent summary."""
    domain_scores = {}
    for domain, features in bundle["domains"].items():
        row = pd.DataFrame([{f: person.get(f) for f in features}])
        domain_scores[domain] = bundle["domain_models"][domain].predict_proba(row)[0, 1]
    meta_row = pd.DataFrame([domain_scores])
    probability = float(bundle["orchestrator"].predict_proba(meta_row)[0, 1])

    # LinearExplainer calculates Shapley values for the actual fitted final model.
    import shap
    background = bundle["background_scores"]
    explainer = shap.LinearExplainer(bundle["orchestrator"], background)
    values = np.asarray(explainer.shap_values(meta_row)).reshape(-1)
    contributions = pd.DataFrame({
        "Subdomain": list(domain_scores),
        "Mini risk score": [domain_scores[d] for d in domain_scores],
        "SHAP contribution": values,
    }).sort_values("SHAP contribution", key=lambda s: s.abs(), ascending=False)
    return probability, contributions, explanation_agent(probability, contributions)


def explanation_agent(probability: float, contributions: pd.DataFrame) -> str:
    """A transparent rule-based explanation agent: it only reads model outputs."""
    level = "high" if probability >= 0.70 else "moderate" if probability >= 0.35 else "low"
    positives = contributions[contributions["SHAP contribution"] > 0].head(3)
    negatives = contributions[contributions["SHAP contribution"] < 0].head(2)
    drivers = ", ".join(positives["Subdomain"].tolist()) or "no subdomain strongly increased the score"
    protectors = ", ".join(negatives["Subdomain"].tolist()) or "no subdomain strongly reduced the score"
    return (
        f"The ensemble estimates a {level} diabetes-risk probability of {probability:.1%}. "
        f"The main risk-increasing subdomains are {drivers}. "
        f"The strongest risk-reducing signals are {protectors}. "
        "This is a model explanation, not a clinical diagnosis; consult a qualified clinician for medical advice."
    )
