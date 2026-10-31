# A Subdomain-Aware Stacked Logistic-Regression Classifier for Diabetes

## Overview

This research implementation uses independent domain-specific logistic-regression models to produce subdomain scores, then combines those scores with a logistic-regression orchestrator. SHAP explains the orchestrator's output, and a transparent rule-based explanation agent turns those explanations into text; no generative language model is used. The NHANES target is self-reported doctor-diagnosed diabetes (`DIQ010`).

## Research-use-only disclaimer

This software is for research and demonstration only. It is not a clinical diagnostic tool, risk model, prevalence estimate, causal model, or validated decision-support system. Its target is self-reported doctor-diagnosed diabetes (`DIQ010`), not a diagnosis established by this software. Do not use its outputs to make clinical decisions.

## Repository structure

- `.gitignore` — excludes Python caches, virtual environments, local secrets, and editor files.
- `CITATION.cff` — citation metadata for the software.
- `LICENSE` — MIT license.
- `README.md` — project, data, installation, and usage documentation.
- `app.py` — Streamlit interface for the synthetic and NHANES model artifacts.
- `diabetes_ensemble.joblib` — included synthetic-model artifact.
- `explain.py` — SHAP contribution calculation and rule-based explanation agent.
- `prepare_nhanes.py` — merges selected NHANES XPT files into the adult analysis CSV.
- `requirements.txt` — pinned direct Python dependencies.
- `train_model.py` — trains the synthetic-data ensemble.
- `train_nhanes_model.py` — trains the NHANES ensemble.
- `nhanes_2017_2018/README.md` — NHANES-specific data and model notes.
- `nhanes_2017_2018/nhanes_diabetes_analysis.csv` — prepared NHANES analysis data.
- `nhanes_2017_2018/nhanes_ensemble.joblib` — trained NHANES model artifact.
- `nhanes_2017_2018/raw/ALQ_J.XPT` — NHANES alcohol questionnaire data.
- `nhanes_2017_2018/raw/BMX_J.XPT` — NHANES body-measure data.
- `nhanes_2017_2018/raw/BPQ_J.XPT` — NHANES blood-pressure questionnaire data.
- `nhanes_2017_2018/raw/BPX_J.XPT` — NHANES blood-pressure examination data.
- `nhanes_2017_2018/raw/DEMO_J.XPT` — NHANES demographic data.
- `nhanes_2017_2018/raw/DIQ_J.XPT` — NHANES diabetes questionnaire data.
- `nhanes_2017_2018/raw/DR1TOT_J.XPT` — NHANES first-day dietary totals.
- `nhanes_2017_2018/raw/GHB_J.XPT` — NHANES glycohemoglobin data.
- `nhanes_2017_2018/raw/GLU_J.XPT` — NHANES fasting glucose data.
- `nhanes_2017_2018/raw/HDL_J.XPT` — NHANES HDL cholesterol data.
- `nhanes_2017_2018/raw/INS_J.XPT` — NHANES insulin data.
- `nhanes_2017_2018/raw/MCQ_J.XPT` — NHANES medical conditions questionnaire data.
- `nhanes_2017_2018/raw/PAQ_J.XPT` — NHANES physical activity questionnaire data.
- `nhanes_2017_2018/raw/SLQ_J.XPT` — NHANES sleep questionnaire data.
- `nhanes_2017_2018/raw/SMQ_J.XPT` — NHANES smoking questionnaire data.
- `nhanes_2017_2018/raw/TCHOL_J.XPT` — NHANES total cholesterol data.
- `nhanes_2017_2018/raw/TRIGLY_J.XPT` — NHANES triglycerides and LDL data.

## Data

The NHANES 2017–2018 public data and documentation are available from the [CDC NHANES continuous survey](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017). The supplied XPT files are in `nhanes_2017_2018/raw/`; `prepare_nhanes.py` creates the analysis CSV from them. The prepared target is self-reported doctor-diagnosed diabetes (`DIQ010`). The analysis is limited to adults and does not apply the NHANES survey design for population estimates.

The intended synthetic benchmark is artificial and contains no real individuals. No synthetic dataset CSV or generator is present in this repository. `train_model.py` requires a separately supplied CSV with the columns defined by its domain feature lists and `diagnosed_diabetes` target; the script does not generate data.

## Installation

From the repository root, create and activate a virtual environment, then install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Usage

Run these commands from the repository root:

```powershell
# Prepare the NHANES analysis dataset from the included raw XPT files.
python prepare_nhanes.py

# Train the synthetic model; first supply the synthetic CSV at this path.
python train_model.py --data "data\Diabetes_and_LifeStyle_Dataset .csv"

# Train the NHANES model from the prepared dataset.
python train_nhanes_model.py

# Launch the Streamlit app.
streamlit run app.py
```

`train_model.py` accepts a caller-provided CSV path through `--data`; it does not create the synthetic input. The app loads the included model artifacts from the repository root. The research-visualization panel also needs its corresponding dataset CSV.

## Known limitation

The orchestrator is trained using domain-model predictions on the same training rows used to fit each domain model, rather than out-of-fold predictions. This can make the orchestrator's training inputs more optimistic than the predictions it receives on unseen data.

## License

This project is released under the MIT License; see `LICENSE`.

## Citation

Please use the metadata in [`CITATION.cff`](CITATION.cff). The paper DOI should be added there after publication.
