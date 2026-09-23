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
    "Predict customer churn individually or upload a CSV file "
    "to predict multiple customers at once."
)

st.divider()


# ==================================================
# CSV BULK PREDICTION
# ==================================================

st.header("📂 Upload Customer Data")

st.write(
    "Upload a CSV file containing customer information "
    "to predict churn for multiple customers."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    # Read uploaded CSV
    uploaded_data = pd.read_csv(uploaded_file)

    st.subheader("📋 Uploaded Customer Data")

    st.dataframe(
        uploaded_data,
        use_container_width=True
    )


    # --------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------

    required_columns = [

        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges"

    ]


    # Check missing columns

    missing_columns = [
        column
        for column in required_columns
        if column not in uploaded_data.columns
    ]


    if missing_columns:

        st.error(
            "❌ Your CSV is missing these required columns: "
            + ", ".join(missing_columns)
        )

    else:

        st.success("✅ CSV format is correct!")


        # --------------------------------------------------
        # BULK PREDICTION
        # --------------------------------------------------

        if st.button(
            "🔮 Predict Uploaded Customers",
            use_container_width=True
        ):

            try:

                # Select only columns required by model
                prediction_data = uploaded_data[
                    required_columns
                ]


                # Make predictions
                predictions = model.predict(
                    prediction_data
                )


                # Get probabilities
                probabilities = model.predict_proba(
                    prediction_data
                )


                # Add prediction results
                uploaded_data["Prediction"] = predictions


                uploaded_data["Churn Probability (%)"] = (
                    probabilities[:, 1] * 100
                ).round(2)


                uploaded_data["Stay Probability (%)"] = (
                    probabilities[:, 0] * 100
                ).round(2)


                # --------------------------------------------------
                # RESULTS
                # --------------------------------------------------

                st.subheader("📊 Prediction Results")

                st.dataframe(
                    uploaded_data,
                    use_container_width=True
                )


                # --------------------------------------------------
                # SUMMARY
                # --------------------------------------------------

                total_customers = len(
                    uploaded_data
                )

                churned_customers = (
                    uploaded_data["Prediction"]
                    == "Yes"
                ).sum()

                stayed_customers = (
                    uploaded_data["Prediction"]
                    == "No"
                ).sum()


                churn_percentage = (
                    churned_customers
                    / total_customers
                ) * 100


                st.subheader(
                    "📈 Churn Summary"
                )


                col1, col2, col3, col4 = st.columns(4)


                with col1:

                    st.metric(
                        "Total Customers",
                        total_customers
                    )


                with col2:

                    st.metric(
                        "Likely to Churn",
                        churned_customers
                    )


                with col3:

                    st.metric(
                        "Likely to Stay",
                        stayed_customers
                    )


                with col4:

                    st.metric(
                        "Churn Rate",
                        f"{churn_percentage:.2f}%"
                    )


                # --------------------------------------------------
                # DOWNLOAD RESULTS
                # --------------------------------------------------

                csv = uploaded_data.to_csv(
                    index=False
                )


                st.download_button(

                    label="⬇️ Download Prediction Results",

                    data=csv,

                    file_name="customer_churn_predictions.csv",

                    mime="text/csv",

                    use_container_width=True
                )


            except Exception as e:

                st.error(
                    f"❌ Prediction failed: {e}"
                )


st.divider()


# ==================================================
# INDIVIDUAL CUSTOMER PREDICTION
# ==================================================

st.header("👤 Individual Customer Prediction")

st.write(
    "Enter information for one customer "
    "to predict whether they are likely to churn."
)


# --------------------------------------------------
# CUSTOMER INFORMATION
# --------------------------------------------------

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
# INDIVIDUAL PREDICTION
# --------------------------------------------------

st.divider()

if st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
):

    try:

        # Make prediction
        prediction = model.predict(
            input_data
        )[0]


        # Get prediction probabilities
        probability = model.predict_proba(
            input_data
        )[0]


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
            min(
                int(churn_probability),
                100
            )
        )


    except Exception as e:

        st.error(
            f"❌ Prediction failed: {e}"
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Customer Churn Prediction System | "
    "Machine Learning using Logistic Regression"
)