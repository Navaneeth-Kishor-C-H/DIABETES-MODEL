# NHANES 2017–2018 real-world data

`raw/` contains public files from the CDC NHANES 2017–2018 release. Run commands from the project root:

```powershell
python prepare_nhanes.py
```

This creates `nhanes_2017_2018/nhanes_diabetes_analysis.csv`, merged by `SEQN` for adults age 18+.

## Train the real-world model

From the project root, run:

```powershell
python train_nhanes_model.py
```

This produces `nhanes_ensemble.joblib` and does not replace the synthetic-data model. The model has eight active subdomains; Family History is transparently omitted because no direct matching variable is available in the selected NHANES cycle.

## Included files

- Clinical: body measures, blood pressure, diabetes questionnaire, cardiovascular questionnaire, glucose, HbA1c, insulin, and lipid panels.
- Lifestyle/social: physical activity, smoking, alcohol, sleep, first-day dietary totals, demographics, education, and income.

## Differences from the synthetic dataset

NHANES does **not** provide direct equivalents for every synthetic field. In particular, it does not provide postprandial glucose, screen time, a single diet score, employment status, or a direct family-history-of-diabetes variable in this selected cycle. The preparation script preserves available real variables rather than inventing replacements. `diagnosed_diabetes` is self-reported doctor-diagnosed diabetes (`DIQ010`), not a clinical diagnosis inferred from laboratory values.

For a publishable NHANES analysis, use the cycle-specific survey weights, strata and PSU variables; consult the [CDC analytic guidelines](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017).
