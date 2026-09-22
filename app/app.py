import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    # Get the project root directory
    project_root = Path(__file__).resolve().parent.parent

    # Path to saved model
    model_path = project_root / "models" / "logistic_regression_model.pkl"

    # Load model
    model = joblib.load(model_path)

    return model


model = load_model()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer information below to predict whether "
    "the customer is likely to churn."
)

st.divider()


# --------------------------------------------------
# CUSTOMER INFORMATION
# --------------------------------------------------

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )


with col2:

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12,
        step=1
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )


with col3:

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


# --------------------------------------------------
# SERVICES
# --------------------------------------------------

st.header("🌐 Services")

col1, col2, col3 = st.columns(3)


with col1:

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


with col2:

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


with col3:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# --------------------------------------------------
# BILLING INFORMATION
# --------------------------------------------------

st.header("💳 Billing Information")

col1, col2 = st.columns(2)


with col1:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=150.0,
        value=70.0,
        step=1.0
    )


with col2:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=float(monthly_charges * tenure),
        step=1.0
    )


payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


# --------------------------------------------------
# CREATE INPUT DATA
# --------------------------------------------------

input_data = pd.DataFrame({

    "gender": [gender],

    "SeniorCitizen": [senior_citizen],

    "Partner": [partner],

    "Dependents": [dependents],

    "tenure": [tenure],

    "PhoneService": [phone_service],

    "MultipleLines": [multiple_lines],

    "InternetService": [internet_service],

    "OnlineSecurity": [online_security],

    "OnlineBackup": [online_backup],

    "DeviceProtection": [device_protection],

    "TechSupport": [tech_support],

    "StreamingTV": [streaming_tv],

    "StreamingMovies": [streaming_movies],

    "Contract": [contract],

    "PaperlessBilling": [paperless_billing],

    "PaymentMethod": [payment_method],

    "MonthlyCharges": [monthly_charges],

    "TotalCharges": [total_charges]
})


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.divider()

if st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
):

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get prediction probabilities
    probability = model.predict_proba(input_data)[0]

    # Probability of churn
    churn_probability = probability[1] * 100

    # Probability of staying
    stay_probability = probability[0] * 100


    # --------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------

    if prediction == "Yes":

        st.error(
            "⚠️ Customer is likely to CHURN"
        )

    else:

        st.success(
            "✅ Customer is likely to STAY"
        )


    # --------------------------------------------------
    # DISPLAY PROBABILITIES
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Churn Probability",
            f"{churn_probability:.2f}%"
        )

    with col2:

        st.metric(
            "Stay Probability",
            f"{stay_probability:.2f}%"
        )


    # Progress bar
    st.write("Churn Risk")

    st.progress(
        min(int(churn_probability), 100)
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Customer Churn Prediction System | "
    "Machine Learning using Logistic Regression"
)