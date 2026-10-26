"""Churn prediction logic (no Streamlit here, so it can be tested on its own)."""
import joblib
import numpy as np
import pandas as pd

MODEL_FILE = "churn_model.pkl"
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]


def load_bundle(path=MODEL_FILE):
    return joblib.load(path)


def predict_churn(new_df, bundle):
    """new_df: rows with the original columns (no Churn, no customerID)."""
    d = new_df.copy()
    d["NumAddOns"] = (d[ADDONS] == "Yes").sum(axis=1)
    d["AvgMonthlySpend"] = np.where(d["tenure"] > 0, d["TotalCharges"] / d["tenure"], 0)
    proba = bundle["model"].predict_proba(d[bundle["columns"]])[:, 1]
    return pd.DataFrame({"churn_probability": proba,
                         "at_risk": proba >= bundle["threshold"]}, index=new_df.index)


def risk_band(p, threshold):
    if p >= 0.60:
        return "High"
    if p >= threshold:
        return "Medium"
    return "Low"
