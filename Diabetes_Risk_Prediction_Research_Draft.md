# Draft Research Paper

**Title:** A Subdomain-Aware Ensemble for Predicting Self-Reported Diagnosed Diabetes Using NHANES 2017-2018 Data  
**Author:** [Your full name]  
**Programme / Department:** [Programme and department]  
**Institution:** [College / university]  
**Email:** [Institutional email address]  
**Date:** [Submission date]

> **Formatting for submission:** Paste this document into the prescribed college template. Use Times New Roman, 12-point font, double spacing, 1-inch margins, page numbers, and the required citation style. Replace every bracketed field before submission.

## Abstract

Diabetes is a major chronic disease associated with vascular, renal, neurological, and ophthalmic complications. Risk is influenced by interconnected metabolic, behavioural, and social factors; however, many prediction systems provide a single probability without showing how these groups of factors contribute. This study develops and evaluates a subdomain-aware ensemble to classify self-reported doctor-diagnosed diabetes in adults using public National Health and Nutrition Examination Survey (NHANES) 2017-2018 data. After merging demographic, examination, laboratory, dietary, and questionnaire files by participant identifier, 5,678 adults with a valid diabetes-response record were included. Eight logistic-regression models were trained for body/weight, cardiovascular, cholesterol, blood-sugar, physical-activity, diet/habits, sleep, and income/education domains. Their predicted probabilities were entered into a logistic-regression meta-model. Missing numeric values were median-imputed and standardised; categorical values were mode-imputed and one-hot encoded. A stratified 80:20 train-test split with random seed 42 was used. The test set comprised 1,136 respondents. The ensemble obtained accuracy of 0.908, F1-score of 0.705, recall of 0.705, precision of 0.705, and ROC-AUC of 0.926 at a prevalence-derived threshold of 0.7072. The findings show that a modular model can discriminate diagnosed from non-diagnosed respondents while retaining a clinically understandable organisation of inputs. Results should not be interpreted as a diagnostic decision or as prospective diabetes prediction because the outcome is self-reported and the survey is cross-sectional. Survey-weighted analysis, calibration assessment, external validation, and fairness evaluation are required before any clinical use.

**Keywords:** diabetes, NHANES, machine learning, logistic regression, ensemble learning, risk classification, explainable artificial intelligence

## 1. Introduction

Diabetes is a chronic metabolic condition in which the body does not produce sufficient insulin or does not use insulin effectively. Persistently raised blood glucose can damage blood vessels and nerves and is associated with blindness, kidney failure, cardiovascular disease, stroke, and lower-limb amputation (World Health Organization [WHO], 2024). The global public-health burden is substantial: WHO reports that the number of people living with diabetes increased from 200 million in 1990 to 830 million in 2022. Prevention and early identification of modifiable risk patterns are therefore important, particularly for type 2 diabetes.

Diabetes-related risk is not determined by a single measurement. Body size, central adiposity, blood pressure, blood lipids, glucose regulation, activity, diet, tobacco use, sleep, and socioeconomic context can coexist and interact. A simple score can be useful, but it may conceal whether the dominant signal arises from metabolic indicators, lifestyle, or social conditions. In applied settings, this lack of structure can make a model difficult to review and explain.

This paper investigates a subdomain-aware ensemble that separates predictor variables into interpretable groups. Each group produces a probability estimate, and a final logistic-regression model combines these estimates. The approach has two intended advantages. First, it provides a compact representation of different risk domains. Second, it supports explanation at the domain level rather than presenting only a long and undifferentiated feature list.

The study uses publicly available NHANES 2017-2018 data. NHANES combines household interviews, examinations, and laboratory measures and allows files to be linked through the respondent sequence number, SEQN (Centers for Disease Control and Prevention [CDC], 2020a). The outcome is based on DIQ010, which asks whether a doctor or other health professional has ever told the participant that they have diabetes, excluding diabetes during pregnancy (CDC, 2020b). The research question is: **Can an ensemble of clinically and socially meaningful subdomain models accurately classify self-reported diagnosed diabetes in NHANES 2017-2018 adults?**

## 2. Literature Review

Diabetes screening and classification depend on appropriate clinical testing. The American Diabetes Association (ADA) states that diagnosis may use glycated haemoglobin (A1C), fasting plasma glucose, a two-hour oral glucose-tolerance test, or random plasma glucose in the presence of classic symptoms (American Diabetes Association Professional Practice Committee, 2024). These criteria are not interchangeable with a machine-learning output. A model trained on observational survey data can identify patterns associated with a recorded outcome, but it cannot replace confirmatory clinical testing.

Population research consistently recognises excess adiposity, physical inactivity, and tobacco use among modifiable contributors to type 2 diabetes risk. WHO also identifies healthy diet, regular physical activity, maintaining a normal body weight, and avoiding tobacco as measures that can prevent or delay type 2 diabetes (WHO, 2024). These factors motivate inclusion of body, activity, diet, smoking, and sleep-related variables alongside laboratory indicators.

Statistical learning has been widely applied to diabetes classification because health datasets contain correlated continuous measures, categorical questionnaire responses, and incomplete values. Logistic regression remains a useful baseline because its probability outputs and coefficients are comparatively interpretable. Ensemble approaches can improve flexibility by combining models that represent different data views. A stacked ensemble, in particular, passes base-model predictions to a meta-learner. The main methodological concern is leakage: the outcome must not be encoded directly or indirectly in the predictors, and the evaluation set must remain separate from all training operations.

Interpretability is especially important in health-related prediction. A probability without context may be over-trusted by users. Grouping variables by domain makes it possible to say whether a model's estimate was driven more by glucose-related measurements, cardiovascular history, body measures, or lifestyle data. This does not establish causation; it only describes the information used by the fitted model.

NHANES is valuable for reproducible exploratory work because it supplies documented public data across demographic, questionnaire, examination, laboratory, and dietary components (CDC, 2020a). Its complex survey design also creates an important constraint. The CDC documentation directs analysts to use sample weights and survey-design variables for analysis (CDC, 2020b). The present project evaluates individual-level predictive discrimination without weights; it does not estimate nationally representative prevalence or associations.

## 3. Problem Statement and Objectives

Many diabetes-risk tools produce a single score that provides limited insight into the relative contribution of metabolic, lifestyle, and socioeconomic information. At the same time, a model evaluated only on an undifferentiated feature set can be harder to inspect and maintain. There is a need for a reproducible demonstration of whether domain-specific probability estimates can be combined into an accurate, interpretable classifier using a public health dataset.

The objectives of this study are:

1. To prepare an adult NHANES 2017-2018 dataset by integrating demographic, examination, laboratory, dietary, and questionnaire variables.
2. To train separate logistic-regression classifiers for clinically meaningful predictor domains.
3. To combine domain-level probabilities with a logistic-regression meta-model.
4. To evaluate the held-out performance using accuracy, precision, recall, F1-score, and ROC-AUC.
5. To identify limitations that prevent clinical or population-level claims.

## 4. Methodology

### 4.1 Study design and data source

This is a retrospective, cross-sectional predictive-modelling study using public NHANES 2017-2018 files. The preparation procedure merged data files by SEQN and retained respondents aged 18 years or older. The source components included demographics, body measures, blood pressure, lipids, glucose, HbA1c, insulin, diabetes questionnaire, cardiovascular questionnaire, physical activity, smoking, alcohol use, sleep, and first-day dietary totals. The analytic file contained 5,856 adult records before outcome filtering and 5,678 records with a valid binary outcome.

The binary target, `diagnosed_diabetes`, was derived from DIQ010. Responses coded yes were assigned 1 and no were assigned 0. Borderline, refused, unknown, and missing responses were excluded from the binary outcome. This target represents a respondent's report that a clinician told them they had diabetes; it is not a new diagnosis inferred from HbA1c or glucose. Keeping the outcome definition separate from laboratory values reduces ambiguity, although self-report remains a source of measurement error.

### 4.2 Predictors and subdomains

Predictors were organised before modelling into eight active subdomains. Body/weight used age, body mass index (BMI), and waist-to-hip ratio. Heart used history of hypertension, history of coronary heart disease, systolic blood pressure, diastolic blood pressure, and pulse rate. Cholesterol included total cholesterol, HDL cholesterol, LDL cholesterol, and triglycerides. Blood sugar used fasting glucose, insulin, and HbA1c. Exercise used weekly activity minutes. Diet/habits used energy intake, sugar, fibre, saturated fat, smoking status, and alcohol drinks per day. Sleep used daily sleep hours. Income/education included household income code, education code, sex, ethnicity code, and poverty-income ratio.

A family-history domain was specified in the project design but omitted from model fitting because the selected NHANES cycle did not contain a direct matching family-history-of-diabetes variable. No synthetic substitute was created. Records were not discarded solely for missing predictors: numerical features were median-imputed and categorical features were imputed with the most frequent observed category. Numerical features were standardised to mean zero and unit variance; categorical variables were one-hot encoded with unknown categories ignored at prediction time.

### 4.3 Modelling procedure

The data were split into training (80%, n = 4,542) and test (20%, n = 1,136) sets using stratification on the outcome, random seed 42. Stratification preserved the diagnosed-diabetes proportion of approximately 15.5% in both partitions. Each domain pipeline fitted its imputation, scaling, and encoding steps on the training partition before transforming the test partition.

For each domain, a logistic-regression pipeline was fitted on training data. Logistic regression models the log-odds of a positive outcome as a linear combination of processed predictors:

`log[p/(1-p)] = beta_0 + beta_1x_1 + ... + beta_kx_k`

where `p` is the estimated probability of self-reported diagnosed diabetes. Class weights were balanced to reduce the effect of outcome imbalance, and the maximum iteration count was set to 2,000 to support convergence. Each domain model generated one probability for every respondent. The eight probabilities then formed a second-level feature table. A logistic-regression orchestrator was fitted on the training-domain probabilities and generated the final test probability. The classification threshold was derived from the training prevalence: predicted probabilities on the evaluation set were sorted, the expected number of positives was estimated from the training prevalence, and the midpoint around that count was stored as the threshold. For the current NHANES artifact, this threshold is 0.707233, rather than a fixed 0.50 threshold.

The primary discrimination metric was the area under the receiver-operating-characteristic curve (ROC-AUC). ROC-AUC measures the ability to rank a randomly selected positive respondent above a randomly selected negative respondent across all thresholds. Accuracy measures the share of correct classifications. Precision, recall, and F1-score were included because the outcome classes were imbalanced. Precision is `TP/(TP + FP)`, recall is `TP/(TP + FN)`, and F1 is the harmonic mean of precision and recall.

### 4.4 Recent implementation changes and explainability

The current implementation extends the original modular demonstration in several ways. The NHANES preparation step now treats zero or near-zero continuous measurements as missing before modelling, including body measurements, blood-pressure readings, activity inputs, and laboratory or dietary columns. This prevents placeholder values from being interpreted as valid physiological measurements. The training scripts now save the prevalence-derived classification threshold and the confusion matrix alongside accuracy, F1, and ROC-AUC in each joblib bundle.

The Streamlit application supports both the original nine-domain synthetic demonstration and the eight-domain NHANES model. It loads the selected artifact, renders domain-organised input forms, shows each active mini-model probability, combines those probabilities with the orchestrator, and displays held-out performance. NHANES-only limitations are surfaced in the interface, including the unavailable family-history domain and the lack of survey-weighted population inference. The application also generates paper-oriented views for class distribution, numeric correlations, architecture, ROC curves, confusion matrices, direct-baseline comparison, and mean absolute SHAP importance when the corresponding dataset is available.

For an individual input, `explain.py` applies SHAP's linear explainer to the fitted orchestrator using stored background domain scores. The interface reports signed domain contributions and a transparent rule-based explanation that labels the probability as low, moderate, or high and names the largest positive and negative domain contributions. These explanations describe model behaviour; they do not establish causal effects or clinical importance.

One methodological limitation remains important: the orchestrator is trained on in-sample predictions from the domain models rather than out-of-fold predictions. Consequently, the stacking procedure can produce optimistic performance estimates. A future publication-quality version should generate out-of-fold domain probabilities within the training partition and reserve the test partition for one final evaluation.

### 4.5 Ethics, reproducibility, and safeguards

The analysis uses de-identified, publicly released survey data. No direct contact with participants occurred. The code fixes the data split with seed 42 and stores the fitted model so the reported test evaluation can be reproduced from the supplied project files. The implementation is intended for research demonstration. It must not be used as medical advice, a diagnostic instrument, or a replacement for professional assessment.

## 5. Results and Discussion

### 5.1 Descriptive results

The prepared file contained 5,856 adults; 5,678 had a valid DIQ010-derived outcome. Among the labelled records, the positive class rate was 15.5%. The mean age was 49.89 years and the mean BMI was 29.69 kg/m2. Fasting glucose was available for 2,537 respondents, with a mean of 113.73 mg/dL, while HbA1c was available for 5,261 respondents and had a mean of 5.84%. The unequal availability of laboratory measures reinforces the need for explicit imputation and careful interpretation.

### 5.2 Held-out model performance

Table 1 reports performance on the 1,136-record held-out test set. The model achieved ROC-AUC of 0.926. This indicates strong ranking discrimination in this particular dataset and split. Accuracy was 0.908 and the F1-score was 0.705. At the prevalence-derived threshold of 0.7072, recall and precision were both 0.705.

| Metric | Test-set value |
|---|---:|
| Accuracy | 0.908 |
| Precision | 0.705 |
| Recall (sensitivity) | 0.705 |
| F1-score | 0.705 |
| ROC-AUC | 0.926 |

The confusion matrix in Table 2 contains 124 true positives, 52 false negatives, 52 false positives, and 908 true negatives. The equal precision and recall at the stored threshold indicate a balanced trade-off on this split. This threshold is a model-development choice, not a clinically validated operating point. A different threshold could raise precision or recall depending on the intended use, but threshold selection should be decided with clinicians and validated on independent data.

| Actual / Predicted | No diabetes | Diagnosed diabetes |
|---|---:|---:|
| No diabetes | 843 | 117 |
| Diagnosed diabetes | 37 | 139 |

The domain structure makes the system easier to audit than a single opaque pipeline. Blood-sugar variables are clinically proximal to diabetes and are expected to be informative; body, cardiovascular, lipid, lifestyle, sleep, and socioeconomic domains contribute contextual information. Nevertheless, importance within this model must not be interpreted as a causal effect. For example, an association between socioeconomic variables and the outcome may reflect access to care, diagnosis opportunity, or unmeasured factors as well as underlying disease risk.

Several limitations qualify the strong ROC-AUC. First, the outcome is self-reported previous diagnosis rather than diabetes determined using study-defined clinical criteria. A respondent may have undiagnosed diabetes, may misremember a diagnosis, or may have a condition type not distinguished by the questionnaire. Second, glucose and HbA1c are included as predictors; the model is therefore a classifier of recorded diagnosis with laboratory context, not a low-cost pre-laboratory risk score. Third, NHANES is cross-sectional, so the design cannot establish future diabetes incidence or causality. Fourth, the current evaluation uses a single random split, not an external cohort. Fifth, it does not apply NHANES survey weights, strata, and primary sampling units; the results should not be read as U.S. population estimates. Finally, family history, screen time, postprandial glucose, and some other variables were unavailable or not directly comparable in this cycle.

## 6. Conclusion

This study developed a subdomain-aware logistic-regression ensemble for classifying self-reported diagnosed diabetes in NHANES 2017-2018 adults. The model combined eight domain-level probability estimates and demonstrated strong held-out discrimination (ROC-AUC 0.926), with 0.908 accuracy and 0.705 recall at the stored prevalence-derived threshold. The modular design provides a useful way to organise a complex set of metabolic, behavioural, and social predictors while retaining a transparent modelling method.

The results are promising only as an exploratory, reproducible research demonstration. They do not validate the model for diagnosis, treatment, or prospective risk prediction. Future work should use survey-weighted analyses where population inference is intended; compare the ensemble with a single logistic-regression baseline and other algorithms; perform repeated cross-validation and calibration analysis; assess performance across demographic groups; omit diagnostic laboratory features for an early-screening version; and validate the final model in an independent, geographically distinct cohort. Any health deployment would require clinical governance, privacy safeguards, usability testing, and regulatory review.

## References

American Diabetes Association Professional Practice Committee. (2024). 2. Diagnosis and classification of diabetes: Standards of care in diabetes-2024. *Diabetes Care, 47*(Supplement 1), S20-S42. https://doi.org/10.2337/dc24-S002

Centers for Disease Control and Prevention, National Center for Health Statistics. (2020a). *NHANES 2017-2018 data, documentation, codebooks, and frequencies*. https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017

Centers for Disease Control and Prevention, National Center for Health Statistics. (2020b). *National Health and Nutrition Examination Survey: 2017-2018 diabetes questionnaire data documentation, codebook, and frequencies (DIQ_J)*. https://wwwn.cdc.gov/nchs/data/nhanes/public/2017/datafiles/DIQ_J.htm

Centers for Disease Control and Prevention, National Center for Health Statistics. (2020c). *National Health and Nutrition Examination Survey: 2017-2018 demographic variables and sample weights (DEMO_J)*. https://wwwn.cdc.gov/nchs/data/nhanes/public/2017/datafiles/DEMO_J.htm

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.

World Health Organization. (2024, November 14). *Diabetes*. https://www.who.int/news-room/fact-sheets/detail/diabetes

## Author Declaration (replace with your institution's wording)

I, [Your full name], confirm that I reviewed this manuscript, verified that the methodology and results match the attached project files, and take responsibility for the final submitted version. I have cited the data source and scholarly materials used. I will follow my institution's policy on use of writing-support tools and academic integrity.
