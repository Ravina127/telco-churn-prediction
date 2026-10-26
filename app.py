import pandas as pd
import streamlit as st

from predict import ADDONS, load_bundle, predict_churn, risk_band

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def get_bundle():
    return load_bundle()


bundle = get_bundle()
threshold = bundle["threshold"]

st.title("📉 Telco Customer Churn Predictor")
st.caption("Enter a customer's details to estimate the chance that they will leave. "
           "Model: Gradient Boosting (ROC-AUC about 0.85 on test data).")

with st.sidebar:
    st.header("About")
    st.write(f"A customer is flagged **at risk** when the churn probability is "
             f"**{threshold:.0%} or higher** (threshold chosen to catch about 64% of churners).")
    st.write("Top churn drivers found in the project: **contract type, tenure, internet service, "
             "tech support and online security**.")
    st.caption("This is a project demo, not a production system.")

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Customer")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Has partner", ["No", "Yes"])
    dependents = st.selectbox("Has dependents", ["No", "Yes"])

with c2:
    st.subheader("Account")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                              "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0, step=0.5)
    total_default = float(round(tenure * monthly, 2))
    total = st.number_input("Total charges (defaults to tenure x monthly)", 0.0, 20000.0,
                            total_default, step=1.0, key=f"total_{tenure}_{monthly}")

with c3:
    st.subheader("Services")
    phone = st.selectbox("Phone service", ["Yes", "No"])
    multiple = st.selectbox("Multiple lines", ["No", "Yes"]) if phone == "Yes" else "No phone service"
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    addon_values = {}
    if internet == "No":
        for a in ADDONS:
            addon_values[a] = "No internet service"
        st.info("No internet service, so add-ons do not apply.")
    else:
        for a in ADDONS:
            addon_values[a] = st.selectbox(a, ["No", "Yes"], key=a)

row = {"gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner,
       "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
       "MultipleLines": multiple, "InternetService": internet, **addon_values,
       "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
       "MonthlyCharges": monthly, "TotalCharges": total}

if st.button("Predict churn risk", type="primary"):
    result = predict_churn(pd.DataFrame([row]), bundle).iloc[0]
    p = float(result["churn_probability"])
    band = risk_band(p, threshold)

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Churn probability", f"{p:.1%}")
    m2.metric("Risk level", band)
    st.progress(min(p, 1.0))

    if band == "High":
        st.error("High risk. Offer a retention incentive soon.")
    elif band == "Medium":
        st.warning("At risk. Consider a check-in call or a contract offer.")
    else:
        st.success("Low risk of churn.")

    tips = []
    if contract == "Month-to-month":
        tips.append("Month-to-month contracts churn the most. Offer a discount for a 1 or 2-year contract.")
    if tenure <= 12:
        tips.append("New customer (first 12 months). Early support and onboarding calls help.")
    if internet == "Fiber optic":
        tips.append("Fibre customers churn more often. Check service quality and pricing.")
    if internet != "No" and addon_values["TechSupport"] == "No":
        tips.append("No tech support. Bundling tech support is linked with lower churn.")
    if internet != "No" and addon_values["OnlineSecurity"] == "No":
        tips.append("No online security. Bundling it is linked with lower churn.")
    if payment == "Electronic check":
        tips.append("Electronic check users churn more. Encourage automatic payment.")
    if tips:
        st.subheader("Possible actions")
        for t in tips:
            st.write("• " + t)
        st.caption("These are patterns from the data, not proven causes.")
