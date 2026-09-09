"""Streamlit interface for the subdomain-aware diabetes-risk ensemble."""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from explain import predict_with_explanation

st.set_page_config(page_title="Diabetes Risk Explorer", page_icon="DR", layout="wide")

st.markdown("""
<style>
    .block-container {max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem;}
    .hero {background: linear-gradient(115deg, #073b4c, #118ab2); color: white;
           border-radius: 18px; padding: 1.8rem 2rem; margin-bottom: 1.4rem;}
    .hero h1 {margin: 0; font-size: 2rem;}
    .hero p {margin: .4rem 0 0; opacity: .9;}
    .domain-note {color: #52706f; font-size: .9rem; margin-top: -.4rem;}
    .disclaimer {background: #fff8e6; color: #17324d; border-left: 4px solid #f4a261; border-radius: 6px; padding: .75rem 1rem; line-height: 1.45;}
    .mode-title {font-size: 1.05rem; font-weight: 700; margin: .3rem 0 -.2rem;}
    div[data-testid="stExpander"] {border-radius: 12px; border-color: rgba(20, 135, 150, .35);}
    div[data-testid="stExpander"] summary {font-weight: 650;}
    div[data-testid="stForm"] {border: 1px solid rgba(20, 135, 150, .35); border-radius: 16px; padding: 1.2rem;}
    div[data-testid="stRadio"] > div {gap: .7rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def build_research_visual_data(model_path, data_path):
    """Recreate the fixed held-out split and prepare paper-figure data."""
    bundle = joblib.load(model_path)
    data = pd.read_csv(data_path)
    target = bundle.get("target", "diagnosed_diabetes")
    data = data[data[target].isin([0, 1])].copy()
    domains = bundle["domains"]
    features = list(dict.fromkeys(f for values in domains.values() for f in values))
    x, y = data[features], data[target].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=42, stratify=y)
    domain_metrics, test_scores = {}, {}
    for domain, cols in domains.items():
        model = bundle["domain_models"][domain]
        probability = model.predict_proba(x_test[cols])[:, 1]
        prediction = (probability >= .5).astype(int)
        domain_metrics[domain] = {"Accuracy": accuracy_score(y_test, prediction), "F1": f1_score(y_test, prediction), "ROC-AUC": roc_auc_score(y_test, probability)}
        test_scores[domain] = probability
    meta_test = pd.DataFrame(test_scores, index=x_test.index)
    final_probability = bundle["orchestrator"].predict_proba(meta_test)[:, 1]
    final_prediction = (final_probability >= .5).astype(int)
    final_metrics = {"Accuracy": accuracy_score(y_test, final_prediction), "F1": f1_score(y_test, final_prediction), "ROC-AUC": roc_auc_score(y_test, final_probability)}

    numeric = [c for c in features if pd.api.types.is_numeric_dtype(x_train[c])]
    categorical = [c for c in features if c not in numeric]
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("fill", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    baseline = Pipeline([("prepare", preprocess), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    baseline.fit(x_train, y_train)
    baseline_probability = baseline.predict_proba(x_test)[:, 1]
    baseline_prediction = (baseline_probability >= .5).astype(int)
    baseline_metrics = {"Accuracy": accuracy_score(y_test, baseline_prediction), "F1": f1_score(y_test, baseline_prediction), "ROC-AUC": roc_auc_score(y_test, baseline_probability)}

    import shap
    shap_values = np.asarray(shap.LinearExplainer(bundle["orchestrator"], bundle["background_scores"]).shap_values(meta_test))
    return {"class_counts": y.value_counts().rename(index={0: "No diabetes", 1: "Diabetes"}), "metrics": {**domain_metrics, "Orchestrator": final_metrics, "Direct baseline": baseline_metrics}, "roc": {"Orchestrator": (y_test.to_numpy(), final_probability), "Direct baseline": (y_test.to_numpy(), baseline_probability)}, "confusion": confusion_matrix(y_test, final_prediction), "correlation": data[numeric].corr(), "shap_importance": pd.Series(np.abs(shap_values).mean(axis=0), index=meta_test.columns).sort_values(ascending=True), "n_test": len(y_test)}


st.markdown("""
<div class="hero">
  <h1>Diabetes Risk Explorer</h1>
  <p>Subdomain-aware risk prediction with transparent SHAP explanations.</p>
</div>
""", unsafe_allow_html=True)
st.markdown("<p class='mode-title'>Choose analysis dataset</p>", unsafe_allow_html=True)
model_choice = st.radio(
    "Choose analysis dataset",
    ["Synthetic demonstration", "Real-world NHANES 2017–2018"],
    horizontal=True,
    label_visibility="collapsed",
)
model_path = "diabetes_ensemble.joblib" if model_choice.startswith("Synthetic") else "nhanes_2017_2018/nhanes_ensemble.joblib"
try:
    bundle = load_bundle(model_path)
except FileNotFoundError:
    st.error(f"Model file missing: {model_path}. Train this model first.")
    st.stop()


DISPLAY = {
    "Age": ("Age", "years"), "bmi": ("Body mass index (BMI)", "kg/m2"),
    "waist_to_hip_ratio": ("Waist-to-hip ratio", "ratio"),
    "hypertension_history": ("History of hypertension", ""),
    "cardiovascular_history": ("History of cardiovascular disease", ""),
    "systolic_bp": ("Systolic blood pressure", "mmHg"),
    "diastolic_bp": ("Diastolic blood pressure", "mmHg"), "heart_rate": ("Resting heart rate", "bpm"),
    "cholesterol_total": ("Total cholesterol", "mg/dL"), "hdl_cholesterol": ("HDL cholesterol", "mg/dL"),
    "ldl_cholesterol": ("LDL cholesterol", "mg/dL"), "triglycerides": ("Triglycerides", "mg/dL"),
    "glucose_fasting": ("Fasting glucose", "mg/dL"), "glucose_postprandial": ("Post-meal glucose", "mg/dL"),
    "insulin_level": ("Insulin level", ""), "hba1c": ("HbA1c", "%"),
    "family_history_diabetes": ("Family history of diabetes", ""),
    "physical_activity_minutes_per_week": ("Physical activity", "minutes/week"),
    "diet_score": ("Diet score", "0-10"), "smoking_status": ("Smoking status", ""),
    "alcohol_consumption_per_week": ("Alcohol consumption", "drinks/week"),
    "screen_time_hours_per_day": ("Screen time", "hours/day"),
    "sleep_hours_per_day": ("Sleep", "hours/day"), "income_level": ("Income level", ""),
    "education_level": ("Education level", ""), "employment_status": ("Employment status", ""),
    "gender": ("Gender", ""), "ethnicity": ("Ethnicity", ""),
    "age": ("Age", "years"), "gender_code": ("Gender code", ""), "ethnicity_code": ("Ethnicity code", ""),
    "education_code": ("Education code", ""), "income_code": ("Income category code", ""),
    "poverty_income_ratio": ("Poverty-income ratio", ""), "waist_cm": ("Waist circumference", "cm"),
    "hip_cm": ("Hip circumference", "cm"), "diet_energy_kcal": ("Dietary energy", "kcal/day"),
    "diet_sugar_g": ("Dietary sugar", "g/day"), "diet_fiber_g": ("Dietary fibre", "g/day"),
    "diet_saturated_fat_g": ("Saturated fat", "g/day"), "alcohol_drinks_per_day": ("Alcohol drinks", "drinks/day"),
}

RANGES = {
    "Age": (18.0, 100.0, 1.0), "bmi": (12.0, 60.0, 0.1), "waist_to_hip_ratio": (0.5, 1.3, 0.01),
    "systolic_bp": (70.0, 220.0, 1.0), "diastolic_bp": (40.0, 140.0, 1.0), "heart_rate": (40.0, 160.0, 1.0),
    "cholesterol_total": (80.0, 400.0, 1.0), "hdl_cholesterol": (10.0, 120.0, 1.0),
    "ldl_cholesterol": (20.0, 300.0, 1.0), "triglycerides": (10.0, 600.0, 1.0),
    "glucose_fasting": (40.0, 400.0, 1.0), "glucose_postprandial": (50.0, 600.0, 1.0),
    "insulin_level": (0.0, 100.0, 0.1), "hba1c": (3.0, 16.0, 0.01),
    "physical_activity_minutes_per_week": (0.0, 1000.0, 1.0), "diet_score": (0.0, 10.0, 0.1),
    "alcohol_consumption_per_week": (0.0, 40.0, 1.0), "screen_time_hours_per_day": (0.0, 24.0, 0.1),
    "sleep_hours_per_day": (0.0, 16.0, 0.1),
    "age": (18.0, 100.0, 1.0), "gender_code": (1.0, 2.0, 1.0), "ethnicity_code": (1.0, 7.0, 1.0),
    "education_code": (1.0, 5.0, 1.0), "income_code": (1.0, 15.0, 1.0), "poverty_income_ratio": (0.0, 6.0, 0.1),
    "diet_energy_kcal": (0.0, 8000.0, 10.0), "diet_sugar_g": (0.0, 500.0, 1.0),
    "diet_fiber_g": (0.0, 100.0, 1.0), "diet_saturated_fat_g": (0.0, 150.0, 1.0),
    "alcohol_drinks_per_day": (0.0, 30.0, 0.1),
}
BINARY = {"hypertension_history", "cardiovascular_history", "family_history_diabetes"}
DOMAIN_NOTES = {
    "Body / Weight": "General body composition indicators.", "Heart": "Blood pressure and cardiovascular history.",
    "Cholesterol": "Lipid profile measurements.", "Blood Sugar": "Glucose regulation and HbA1c.",
    "Family History": "Inherited or familial risk signal.", "Exercise": "Weekly physical activity.",
    "Diet / Habits": "Diet and daily health habits.", "Sleep": "Average sleeping duration.",
    "Income / Education": "Social and demographic context.",
}


def label(feature):
    name, unit = DISPLAY.get(feature, (feature.replace("_", " ").title(), ""))
    return f"{name} ({unit})" if unit else name


def input_for(feature: str) -> object:
    """Render one field using an appropriate input control."""
    if feature in BINARY:
        default = int(bundle["default_values"].get(feature, 0))
        choice = st.selectbox(label(feature), ["No", "Yes"], index=default, key=feature)
        return 1 if choice == "Yes" else 0
    if feature in bundle["category_values"]:
        options = bundle["category_values"][feature]
        return st.selectbox(label(feature), options, key=feature)
    low, high, step = RANGES[feature]
    default = float(bundle["default_values"].get(feature, low))
    default = min(max(default, low), high)
    return st.number_input(label(feature), min_value=low, max_value=high, value=default, step=step, key=feature)


def render_domain(domain: str, person: dict):
    st.markdown(f"#### {domain}")
    st.markdown(f"<p class='domain-note'>{DOMAIN_NOTES.get(domain, 'Model input group.')}</p>", unsafe_allow_html=True)
    features = bundle["domains"][domain]
    columns = st.columns(2)
    for index, feature in enumerate(features):
        with columns[index % 2]:
            person[feature] = input_for(feature)


if model_choice.startswith("Synthetic"):
    notice = "Synthetic-data demonstration. This tool is not a medical diagnosis."
else:
    notice = "Real-world NHANES 2017–2018 validation model. Family History is unavailable in this cycle; results are not a medical diagnosis."
st.markdown(f"<div class='disclaimer'>{notice}</div>", unsafe_allow_html=True)
st.write("")
st.write("Enter participant information below. Each section corresponds directly to one model subdomain.")

person = {}
clinical = [d for d in ["Body / Weight", "Heart", "Cholesterol", "Blood Sugar", "Family History"] if d in bundle["domains"]]
lifestyle = [d for d in ["Exercise", "Diet / Habits", "Sleep", "Income / Education"] if d in bundle["domains"]]

with st.form("risk_form"):
    clinical_tab, lifestyle_tab = st.tabs(["Clinical subdomains", "Lifestyle and social subdomains"])
    with clinical_tab:
        for domain in clinical:
            with st.expander(domain, expanded=(domain == "Blood Sugar")):
                render_domain(domain, person)
    with lifestyle_tab:
        for domain in lifestyle:
            with st.expander(domain, expanded=(domain == "Exercise")):
                render_domain(domain, person)
        if bundle.get("unavailable_domains"):
            st.warning("Unavailable in this NHANES cycle: " + ", ".join(bundle["unavailable_domains"]) + ". It is excluded rather than estimated from unrelated data.")
    submitted = st.form_submit_button("Calculate diabetes risk", type="primary", use_container_width=True)

if submitted:
    risk, contributions, summary = predict_with_explanation(bundle, person)
    st.divider()
    st.subheader("Step 1 and 2: Subdomain mini-models")
    st.write(f"Each card below is the output of one independently trained subdomain model. These {len(bundle['domains'])} probabilities are the inputs to the final orchestrator.")
    st.info("Mini-model probabilities do not add to 100%, because every mini-model independently estimates the same outcome: diabetes risk. The **decision influence** bars below are normalized SHAP contributions and therefore add to 100%.")
    score_lookup = contributions.set_index("Subdomain")["Mini risk score"].to_dict()
    total_influence = contributions["SHAP contribution"].abs().sum()
    influence_lookup = (
        contributions.set_index("Subdomain")["SHAP contribution"].abs() / total_influence
        if total_influence > 0 else pd.Series(1 / len(contributions), index=contributions["Subdomain"])
    ).to_dict()
    for start in range(0, len(bundle["domains"]), 3):
        columns = st.columns(3)
        for column, domain in zip(columns, list(bundle["domains"])[start:start + 3]):
            with column:
                with st.container(border=True):
                    st.caption("Independent mini-model risk")
                    st.metric(domain, f"{score_lookup[domain]:.1%}")
                    influence = influence_lookup[domain]
                    st.progress(int(round(influence * 100)), text=f"Decision influence: {influence:.1%}")
                    st.caption(DOMAIN_NOTES.get(domain, "Model input group."))

    st.markdown(f"<div style='text-align:center; color:#118ab2; font-size:1.5rem; padding:.6rem'>{len(bundle['domains'])} mini-model scores &darr; final orchestrator</div>", unsafe_allow_html=True)
    st.subheader("Step 3: Combined final prediction")
    left, right = st.columns([1, 2])
    with left:
        st.metric("Diabetes-risk probability", f"{risk:.1%}")
        st.caption("The orchestrator combines the nine mini-model probabilities into this single prediction.")
    with right:
        st.subheader("Explanation Agent report")
        st.write(summary)

    st.subheader("Step 4 and 5: SHAP explanation and Explanation Agent")
    st.caption("SHAP values are computed from the final orchestrator. Positive values increase predicted risk; negative values reduce it.")
    chart_data = contributions.set_index("Subdomain")["SHAP contribution"]
    st.bar_chart(chart_data, color="#118ab2")
    table = contributions.copy()
    table["Mini risk score"] = table["Mini risk score"].map("{:.1%}".format)
    table["SHAP contribution"] = table["SHAP contribution"].map("{:+.3f}".format)
    st.dataframe(table, hide_index=True, use_container_width=True)

with st.expander("Model performance and research note"):
    st.write("Held-out test performance")
    metrics = bundle["metrics"]
    a, b, c = st.columns(3)
    a.metric("Accuracy", metrics["accuracy"])
    b.metric("F1 score", metrics["f1"])
    c.metric("ROC-AUC", metrics["roc_auc"])
    st.caption("Synthetic data only; external validation is required." if model_choice.startswith("Synthetic") else "NHANES 2017–2018 real-world validation results. Survey weights and design variables are required for population-representative estimates.")

st.divider()
with st.expander("Research visualizations (paper figures)", expanded=False):
    data_path = (r"C:\Users\navan\Downloads\Diabetes_and_LifeStyle_Dataset .csv"
                 if model_choice.startswith("Synthetic") else "nhanes_2017_2018/nhanes_diabetes_analysis.csv")
    try:
        figures = build_research_visual_data(model_path, data_path)
        overview_tab, architecture_tab, performance_tab, explainability_tab = st.tabs([
            "Dataset overview", "Implementation architecture", "Model comparison", "Explainability"
        ])
        with overview_tab:
            st.subheader("Target distribution")
            st.bar_chart(figures["class_counts"], color="#118ab2")
            st.caption(f"The charts below use the fixed held-out test set ({figures['n_test']:,} records) unless stated otherwise.")
            st.subheader("Numeric feature correlation")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 6))
            image = ax.imshow(figures["correlation"], cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(figures["correlation"].columns)), figures["correlation"].columns, rotation=90, fontsize=7)
            ax.set_yticks(range(len(figures["correlation"].index)), figures["correlation"].index, fontsize=7)
            fig.colorbar(image, ax=ax, fraction=.04, label="Pearson correlation")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
        with architecture_tab:
            st.subheader("Subdomain-aware implementation")
            graph_domains = "\\n".join(bundle["domains"])
            st.graphviz_chart(f'''digraph G {{ rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#e8f5f6"]; input [label="Clinical + lifestyle inputs", fillcolor="#d8f3dc"]; domains [label="Subdomain models\\n{graph_domains}"]; orchestrator [label="Final orchestrator", fillcolor="#ffd166"]; output [label="Risk + SHAP + Explanation Agent", fillcolor="#cde8f6"]; input -> domains -> orchestrator -> output; }}''', use_container_width=True)
            st.write("Each subdomain model generates one probability. The orchestrator receives those probabilities and produces the final prediction.")
            st.subheader("Explanation Agent workflow")
            st.graphviz_chart('''digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#e8f5f6"]; prediction [label="Final prediction"]; shap [label="SHAP values", fillcolor="#ffd166"]; agent [label="Explanation Agent", fillcolor="#cde8f6"]; report [label="Plain-language report"]; prediction -> shap -> agent -> report; }''', use_container_width=True)
        with performance_tab:
            st.subheader("Model comparison")
            metric_frame = pd.DataFrame(figures["metrics"]).T
            st.bar_chart(metric_frame, color=["#118ab2", "#06d6a0", "#ef476f"])
            st.dataframe(metric_frame.style.format("{:.3f}"), use_container_width=True)
            st.caption("The direct baseline uses all available features in one logistic-regression model. Subdomain models are shown to document modular performance; the orchestrator is the proposed model.")
            st.subheader("ROC curve comparison")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for name, (truth, probability) in figures["roc"].items():
                fpr, tpr, _ = roc_curve(truth, probability)
                ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(truth, probability):.3f})")
            ax.plot([0, 1], [0, 1], "k--", alpha=.5, label="Random")
            ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Held-out ROC curves")
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            st.subheader("Orchestrator confusion matrix")
            st.dataframe(pd.DataFrame(figures["confusion"], index=["Actual: no diabetes", "Actual: diabetes"], columns=["Predicted: no diabetes", "Predicted: diabetes"]), use_container_width=True)
        with explainability_tab:
            st.subheader("Average subdomain importance")
            st.bar_chart(figures["shap_importance"], color="#f4a261")
            st.caption("Mean absolute SHAP value on the held-out set. Larger values indicate greater average influence on the orchestrator's internal prediction.")
            st.write("The individual SHAP explanation above the visualizations remains available after submitting a participant profile.")
    except Exception as error:
        st.warning(f"Research figures are unavailable until the selected dataset/model is present and compatible: {error}")
