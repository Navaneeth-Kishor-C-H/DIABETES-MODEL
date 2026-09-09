# Subdomain-Aware Diabetes Risk Prediction

This is a compact implementation of the proposed research model. It trains nine small logistic-regression models (one per clinical/lifestyle subdomain), then passes their risk scores to a final logistic-regression orchestrator. The UI calculates SHAP values on that final model and produces a plain-language explanation from those values.

## Setup

```powershell
$python = "C:\Users\navan\AppData\Local\Programs\Python\Python312\python.exe"
& $python -m pip install -r requirements.txt
& $python train_model.py --data "C:\Users\navan\Downloads\Diabetes_and_LifeStyle_Dataset .csv"
& $python -m streamlit run app.py
```

The explicit Python path is used because the Windows `py -3` launcher is currently linked to a broken Store Python installation on this computer.

The held-out test metrics print during training and appear in the UI. Random seed 42 and an 80/20 stratified split make the run reproducible.

## Important research notes

- The provided dataset is synthetic. State this in the methodology and limitations sections.
- `diabetes_risk_score` and `diabetes_stage` are excluded because they can reveal the diagnosis and create target leakage.
- Treat the output as a research demonstration, not medical advice or a clinical tool.
- Real-world external validation (for example, on NHANES) is needed before claiming generalizability.
