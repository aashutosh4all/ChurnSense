import requests
import streamlit as st

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("Customer Churn Prediction App")
st.write(
    "Enter customer details below to predict churn probability and retention risk."
)

st.divider()

# -----------------------------
# Input Form
# -----------------------------
with st.form("customer_form"):
    st.subheader("Customer Information")

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=45
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=10,
        value=3
    )

    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=120000.0
    )

    num_products = st.selectbox(
        "Number of Products",
        [1, 2, 3, 4]
    )

    has_cr_card_label = st.selectbox(
        "Has Credit Card?",
        ["Yes", "No"]
    )

    is_active_label = st.selectbox(
        "Is Active Member?",
        ["Yes", "No"]
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=100000.0
    )

    submitted = st.form_submit_button("Predict Churn")

# -----------------------------
# Prediction Logic
# -----------------------------
if submitted:
    has_cr_card = 1 if has_cr_card_label == "Yes" else 0
    is_active_member = 1 if is_active_label == "Yes" else 0

    customer_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary
    }

    try:
        response = requests.post(API_URL, json=customer_data)

        if response.status_code == 200:
            result = response.json()

            churn_probability = result["churn_probability_percentage"]
            prediction = result["prediction"]
            risk_level = result["risk_level"]
            threshold = result["threshold_used"]
            suggestion = result["retention_suggestion"]

            st.divider()
            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Churn Probability",
                    value=f"{churn_probability}%"
                )

            with col2:
                st.metric(
                    label="Prediction",
                    value=prediction
                )

            st.write(f"**Risk Level:** {risk_level}")
            st.write(f"**Threshold Used:** {threshold}")

            if prediction == "Churn":
                st.warning("This customer is likely to churn.")
            else:
                st.success("This customer is less likely to churn.")

            st.info(f"Retention Suggestion: {suggestion}")

        else:
            st.error("API request failed. Please check backend server.")

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to FastAPI backend. "
            "Make sure the backend server is running on http://127.0.0.1:8000"
        )