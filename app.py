import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")

MODEL_PATH = "telco_churn_logreg_tuned.pkl"

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

pipe = load_model()
st.success("Model loaded successfully!")

st.title("📞 Telco Customer Churn Predictor")
st.markdown("""
Predict whether a customer is likely to **churn** (leave) based on their profile and services.  
Enter the details below and click **Predict**.
""")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=800.0, step=10.0)
        cltv = st.number_input("CLTV", min_value=0, value=2000)

    with col2:
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

    submit = st.form_submit_button("Predict Churn Risk")

if submit:
    online_security_flag = 1 if online_security == 'Yes' else 0
    online_backup_flag = 1 if online_backup == 'Yes' else 0
    device_protection_flag = 1 if device_protection == 'Yes' else 0
    tech_support_flag = 1 if tech_support == 'Yes' else 0
    streaming_tv_flag = 1 if streaming_tv == 'Yes' else 0
    streaming_movies_flag = 1 if streaming_movies == 'Yes' else 0

    total_services = sum([
        1 if phone_service == 'Yes' else 0,
        1 if multiple_lines == 'Yes' else 0,
        1 if online_security == 'Yes' else 0,
        1 if online_backup == 'Yes' else 0,
        1 if device_protection == 'Yes' else 0,
        1 if tech_support == 'Yes' else 0,
        1 if streaming_tv == 'Yes' else 0,
        1 if streaming_movies == 'Yes' else 0
    ])

    has_addons = 1 if total_services > 2 else 0
    has_fiber = 1 if internet_service == 'Fiber optic' else 0
    has_autopay = 1 if 'automatic' in payment_method else 0
    is_senior = 1 if senior == 'Yes' else 0
    is_family = 1 if partner == 'Yes' or dependents == 'Yes' else 0
    is_new_customer = 1 if tenure < 6 else 0

    if tenure <= 6:
        tenure_group = '0-6'
    elif tenure <= 12:
        tenure_group = '7-12'
    elif tenure <= 24:
        tenure_group = '13-24'
    elif tenure <= 36:
        tenure_group = '25-36'
    elif tenure <= 48:
        tenure_group = '37-48'
    elif tenure <= 60:
        tenure_group = '49-60'
    else:
        tenure_group = '61-72'

    avg_charge = total_charges / tenure if tenure > 0 else monthly_charges

    input_data = {
        'Gender': [gender],
        'Senior Citizen': [senior],
        'Partner': [partner],
        'Dependents': [dependents],
        'Tenure Months': [tenure],
        'Phone Service': [phone_service],
        'Multiple Lines': [multiple_lines],
        'Internet Service': [internet_service],
        'Online Security': [online_security],
        'Online Backup': [online_backup],
        'Device Protection': [device_protection],
        'Tech Support': [tech_support],
        'Streaming TV': [streaming_tv],
        'Streaming Movies': [streaming_movies],
        'Contract': [contract],
        'Paperless Billing': [paperless],
        'Payment Method': [payment_method],
        'Monthly Charges': [monthly_charges],
        'Total Charges': [total_charges],
        'CLTV': [cltv],
        'Tenure_Group': [tenure_group],
        'Online Security_flag': [online_security_flag],
        'Online Backup_flag': [online_backup_flag],
        'Device Protection_flag': [device_protection_flag],
        'Tech Support_flag': [tech_support_flag],
        'Streaming TV_flag': [streaming_tv_flag],
        'Streaming Movies_flag': [streaming_movies_flag],
        'Total_Services': [total_services],
        'Has_Addons': [has_addons],
        'Has_Fiber': [has_fiber],
        'Has_Autopay': [has_autopay],
        'Is_Senior': [is_senior],
        'Is_Family': [is_family],
        'Is_New_Customer': [is_new_customer],
        'Avg_Charge': [avg_charge]
    }

    df_input = pd.DataFrame(input_data)

    try:
        prob = pipe.predict_proba(df_input)[:, 1][0]
        prediction = pipe.predict(df_input)[0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"🔴 **High Risk of Churn** – Probability: **{prob:.1%}**")
        else:
            st.success(f"🟢 **Low Risk – Likely to Stay** – Probability: **{prob:.1%}**")

        st.progress(int(prob * 100))
        st.caption(f"Churn Probability: {prob:.1%}")

    except Exception as e:
        st.error(f"Prediction error: {e}\nMake sure all required columns are present in the input.")

st.markdown("---")
st.caption("Built with Streamlit • Model: Tuned Logistic Regression (AUC ≈ 0.863) • Project by Ganesh Sahu")