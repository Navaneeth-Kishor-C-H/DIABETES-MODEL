"""Create an analysis-ready real-world NHANES 2017-2018 diabetes dataset.

The source XPT files must be in nhanes_2017_2018/raw (already downloaded).
Run: python prepare_nhanes.py

This creates nhanes_2017_2018/nhanes_diabetes_analysis.csv.  It is NOT a
drop-in replacement for the synthetic CSV: some synthetic variables have no
direct NHANES equivalent (post-meal glucose, screen time and family history).
"""
from pathlib import Path

import pandas as pd

RAW = Path("nhanes_2017_2018/raw")
OUT = Path("nhanes_2017_2018/nhanes_diabetes_analysis.csv")


def read(name, columns):
    """Load only needed variables; every NHANES public file shares SEQN."""
    frame = pd.read_sas(RAW / f"{name}.XPT", format="xport", encoding="utf-8")
    return frame[["SEQN", *columns]]


def yes_no(series):
    """NHANES questionnaire codes: 1=yes, 2=no; other codes are missing."""
    return series.map({1.0: 1, 2.0: 0})


def main():
    # Demographics / social context
    demo = read("DEMO_J", ["RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDHHIN2", "INDFMPIR", "WTMEC2YR"])
    demo = demo.rename(columns={
        "RIDAGEYR": "age", "RIAGENDR": "gender_code", "RIDRETH3": "ethnicity_code",
        "DMDEDUC2": "education_code", "INDHHIN2": "income_code", "INDFMPIR": "poverty_income_ratio",
        "WTMEC2YR": "mec_exam_weight",
    })

    body = read("BMX_J", ["BMXBMI", "BMXWAIST", "BMXHIP"]).rename(columns={"BMXBMI": "bmi", "BMXWAIST": "waist_cm", "BMXHIP": "hip_cm"})
    body["waist_to_hip_ratio"] = body["waist_cm"] / body["hip_cm"]

    bp = read("BPX_J", ["BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1", "BPXDI2", "BPXDI3", "BPXPLS"])
    bp["systolic_bp"] = bp[["BPXSY1", "BPXSY2", "BPXSY3"]].mean(axis=1)
    bp["diastolic_bp"] = bp[["BPXDI1", "BPXDI2", "BPXDI3"]].mean(axis=1)
    bp = bp[["SEQN", "systolic_bp", "diastolic_bp", "BPXPLS"]].rename(columns={"BPXPLS": "heart_rate"})

    lipids = read("TCHOL_J", ["LBXTC"]).merge(read("HDL_J", ["LBDHDD"]), on="SEQN", how="outer").merge(read("TRIGLY_J", ["LBXTR", "LBDLDL"]), on="SEQN", how="outer")
    lipids = lipids.rename(columns={"LBXTC": "cholesterol_total", "LBDHDD": "hdl_cholesterol", "LBXTR": "triglycerides", "LBDLDL": "ldl_cholesterol"})

    sugar = read("GLU_J", ["LBXGLU"]).merge(read("GHB_J", ["LBXGH"]), on="SEQN", how="outer").merge(read("INS_J", ["LBXIN"]), on="SEQN", how="outer")
    sugar = sugar.rename(columns={"LBXGLU": "glucose_fasting", "LBXGH": "hba1c", "LBXIN": "insulin_level"})

    # Target: self-reported doctor-diagnosed diabetes.  Restrict to adults later.
    diabetes = read("DIQ_J", ["DIQ010"]).assign(diagnosed_diabetes=lambda x: yes_no(x["DIQ010"]))[["SEQN", "diagnosed_diabetes"]]
    hypertension = read("BPQ_J", ["BPQ020"]).assign(hypertension_history=lambda x: yes_no(x["BPQ020"]))[["SEQN", "hypertension_history"]]
    # MCQ160C is doctor-diagnosed coronary heart disease; this is a defensible cardiovascular-history proxy.
    heart = read("MCQ_J", ["MCQ160C"]).assign(cardiovascular_history=lambda x: yes_no(x["MCQ160C"]))[["SEQN", "cardiovascular_history"]]

    activity = read("PAQ_J", ["PAQ650", "PAD660"])
    activity["physical_activity_minutes_per_week"] = activity["PAQ650"].map({1.0: 1, 2.0: 0}) * activity["PAD660"] * 7
    activity = activity[["SEQN", "physical_activity_minutes_per_week"]]
    smoking = read("SMQ_J", ["SMQ020", "SMQ040"])
    smoking["smoking_status"] = "Unknown"
    smoking.loc[smoking["SMQ020"] == 2, "smoking_status"] = "Never"
    smoking.loc[(smoking["SMQ020"] == 1) & (smoking["SMQ040"] == 3), "smoking_status"] = "Former"
    smoking.loc[(smoking["SMQ020"] == 1) & smoking["SMQ040"].isin([1, 2]), "smoking_status"] = "Current"
    smoking = smoking[["SEQN", "smoking_status"]]
    alcohol = read("ALQ_J", ["ALQ121", "ALQ130"])
    # Frequency code is not directly weekly; retain raw frequency and drinks/day transparently.
    alcohol = alcohol.rename(columns={"ALQ121": "alcohol_frequency_code", "ALQ130": "alcohol_drinks_per_day"})
    sleep = read("SLQ_J", ["SLD012"]).rename(columns={"SLD012": "sleep_hours_per_day"})
    diet = read("DR1TOT_J", ["DR1TKCAL", "DR1TSUGR", "DR1TFIBE", "DR1TSFAT"])
    diet = diet.rename(columns={"DR1TKCAL": "diet_energy_kcal", "DR1TSUGR": "diet_sugar_g", "DR1TFIBE": "diet_fiber_g", "DR1TSFAT": "diet_saturated_fat_g"})

    tables = [demo, body, bp, lipids, sugar, diabetes, hypertension, heart, activity, smoking, alcohol, sleep, diet]
    data = tables[0]
    for table in tables[1:]:
        data = data.merge(table, on="SEQN", how="left")
    data = data[data["age"] >= 18].copy()
    data.to_csv(OUT, index=False)
    print(f"Saved {OUT} with {len(data):,} adult participants and {len(data.columns)} columns.")
    print("Diagnosed-diabetes records:", int(data["diagnosed_diabetes"].notna().sum()))
    print("Note: use NHANES MEC/laboratory weights and survey design variables for population estimates.")


if __name__ == "__main__":
    main()
