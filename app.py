from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Banking Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background:
        linear-gradient(rgba(5, 15, 35, 0.94),
                        rgba(5, 15, 35, 0.94)),
        radial-gradient(circle at top right, #123c6b, #050f23);
    color: white;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06172d, #09284a);
    border-right: 1px solid rgba(100, 170, 255, 0.25);
}

.main-title {
    font-size: 48px;
    font-weight: 750;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    color: #b8cce5;
    font-size: 18px;
    margin-bottom: 28px;
}

.model-card {
    background: rgba(12, 37, 68, 0.90);
    border: 1px solid rgba(100, 170, 255, 0.24);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.30);
}

.result-low {
    background: rgba(18, 110, 75, 0.30);
    border: 1px solid #35d39a;
    border-radius: 14px;
    padding: 20px;
    font-size: 21px;
    text-align: center;
}

.result-high {
    background: rgba(155, 42, 42, 0.30);
    border: 1px solid #ff6b6b;
    border-radius: 14px;
    padding: 20px;
    font-size: 21px;
    text-align: center;
}

.result-info {
    background: rgba(35, 91, 155, 0.30);
    border: 1px solid #5aa9ff;
    border-radius: 14px;
    padding: 20px;
    font-size: 21px;
    text-align: center;
}

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 650;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load saved models
# ---------------------------------------------------------
MODEL_DIR = Path(__file__).parent / "models"


@st.cache_resource
def load_models():
    try:
        return {
            "kmeans": joblib.load(MODEL_DIR / "kmeans_model.pkl"),
            "scaler": joblib.load(MODEL_DIR / "scaler.pkl"),
            "loan": joblib.load(MODEL_DIR / "loan_risk_model.pkl"),
            "card": joblib.load(MODEL_DIR / "card_adoption_model.pkl")
        }
    except FileNotFoundError as error:
        st.error(f"Model file not found: {error}")
        st.stop()
    except Exception as error:
        st.error(f"Unable to load models: {error}")
        st.stop()


models = load_models()


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def align_features(data: pd.DataFrame, model) -> pd.DataFrame:
    """
    Align input columns with the exact columns used during training.
    Missing dummy columns are created with value 0.
    """
    if hasattr(model, "feature_names_in_"):
        return data.reindex(
            columns=list(model.feature_names_in_),
            fill_value=0
        )

    return data


def segment_description(cluster: int) -> tuple[str, str]:
    """
    Cluster numbers depend on model training.
    These are general business labels and may be edited after
    checking the cluster summary in the notebook.
    """
    descriptions = {
        0: (
            "Standard Customer",
            "Maintain regular engagement and recommend suitable basic products."
        ),
        1: (
            "High-Value Customer",
            "Offer premium banking products and personalised services."
        ),
        2: (
            "Active Customer",
            "Cross-sell cards, loans and digital banking services."
        ),
        3: (
            "Low-Activity Customer",
            "Use targeted campaigns and engagement offers."
        )
    }

    return descriptions.get(
        int(cluster),
        ("Customer Segment", "Review this customer's banking behaviour.")
    )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.markdown("## 🏦 Banking Analytics")

selected_model = st.sidebar.radio(
    "Select Prediction Model",
    [
        "Customer Segmentation",
        "Loan Risk Prediction",
        "Card Adoption Prediction"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("""
### Models Used

- **K-Means** — Customer Segmentation  
- **Random Forest** — Loan Risk  
- **Random Forest** — Card Adoption
""")

st.sidebar.caption("Built using Python, Scikit-learn and Streamlit")


# ---------------------------------------------------------
# Main heading
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🏦 Banking Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Interactive customer segmentation, loan-risk and card-adoption predictions.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# MODEL 1: CUSTOMER SEGMENTATION
# =========================================================
if selected_model == "Customer Segmentation":

    st.markdown("""
    <div class="model-card">
        <h2>👥 Customer Segmentation</h2>
        <p>
        Enter customer behaviour details to identify the most suitable
        customer segment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1
        )

        transaction_count = st.number_input(
            "Transaction Count",
            min_value=0,
            value=100,
            step=1
        )

    with col2:
        total_amount = st.number_input(
            "Total Transaction Amount",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        average_balance = st.number_input(
            "Average Balance",
            value=10000.0,
            step=500.0
        )

    if st.button("Predict Customer Segment"):

        customer = pd.DataFrame([{
            "age": age,
            "Transaction_Count": transaction_count,
            "Total_Amount": total_amount,
            "Avg_Balance": average_balance
        }])

        try:
            scaler = models["scaler"]
            kmeans = models["kmeans"]

            customer = align_features(customer, scaler)
            scaled_customer = scaler.transform(customer)

            cluster = int(kmeans.predict(scaled_customer)[0])
            label, recommendation = segment_description(cluster)

            st.markdown(
                f"""
                <div class="result-info">
                    <strong>Predicted Segment:</strong>
                    Cluster {cluster} — {label}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.info(f"Recommendation: {recommendation}")

        except Exception as error:
            st.error(
                "Prediction failed because the app input features do not "
                f"match the saved model. Details: {error}"
            )


# =========================================================
# MODEL 2: LOAN RISK PREDICTION
# =========================================================
elif selected_model == "Loan Risk Prediction":

    st.markdown("""
    <div class="model-card">
        <h2>💰 Loan Risk Prediction</h2>
        <p>
        Estimate whether a loan application belongs to a low-risk
        or high-risk category.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=100000.0,
            step=5000.0
        )

        duration = st.number_input(
            "Duration in Months",
            min_value=1,
            value=24,
            step=1
        )

        monthly_payment = st.number_input(
            "Monthly Payment",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

        customer_age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1
        )

    with col2:
        average_salary = st.number_input(
            "District Average Salary",
            min_value=0.0,
            value=10000.0,
            step=500.0
        )

        unemployment_rate = st.number_input(
            "District Unemployment Rate",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1
        )

        crime_count = st.number_input(
            "District Crime Count",
            min_value=0.0,
            value=3000.0,
            step=100.0
        )

    if st.button("Predict Loan Risk"):

        loan_input = pd.DataFrame([{
            "amount": loan_amount,
            "duration": duration,
            "payments": monthly_payment,
            "age": customer_age,
            "Average_Salary": average_salary,
            "Unemployment_Rate_Current_Year": unemployment_rate,
            "Crimes_Current_Year": crime_count
        }])

        try:
            loan_model = models["loan"]
            loan_input = align_features(loan_input, loan_model)

            prediction = int(loan_model.predict(loan_input)[0])

            if hasattr(loan_model, "predict_proba"):
                probability = float(
                    loan_model.predict_proba(loan_input)[0][1]
                )
            else:
                probability = float(prediction)

            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="result-high">
                        <strong>High-Risk Loan</strong><br>
                        Estimated default probability:
                        {probability:.1%}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.warning(
                    "Recommendation: Conduct additional verification, "
                    "review repayment capacity and apply closer monitoring."
                )

            else:
                st.markdown(
                    f"""
                    <div class="result-low">
                        <strong>Low-Risk Loan</strong><br>
                        Estimated default probability:
                        {probability:.1%}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.success(
                    "Recommendation: The application may proceed through "
                    "the bank's standard approval process."
                )

        except Exception as error:
            st.error(
                "Prediction failed because the app input features do not "
                f"match the saved loan model. Details: {error}"
            )


# =========================================================
# MODEL 3: CARD ADOPTION PREDICTION
# =========================================================
elif selected_model == "Card Adoption Prediction":

    st.markdown("""
    <div class="model-card">
        <h2>💳 Card Adoption Prediction</h2>
        <p>
        Estimate the likelihood that a customer will adopt a bank card.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        card_age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1,
            key="card_age"
        )

        sex = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        disposition_type = st.selectbox(
            "Customer Role",
            ["OWNER", "USER"]
        )

        account_type = st.selectbox(
            "Account Type",
            ["Standard", "Salary", "Business", "Other"]
        )

    with col2:
        frequency = st.selectbox(
            "Statement Frequency",
            ["Monthly Issuance", "Weekly Issuance", "After Transaction"]
        )

        card_transaction_count = st.number_input(
            "Transaction Count",
            min_value=0,
            value=100,
            step=1,
            key="card_transactions"
        )

        card_total_amount = st.number_input(
            "Total Transaction Amount",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            key="card_total"
        )

        card_average_balance = st.number_input(
            "Average Balance",
            value=10000.0,
            step=500.0,
            key="card_balance"
        )

    if st.button("Predict Card Adoption"):

        raw_card_input = pd.DataFrame([{
            "age": card_age,
            "sex": sex,
            "type": disposition_type,
            "Account_type": account_type,
            "frequency": frequency,
            "Transaction_Count": card_transaction_count,
            "Total_Amount": card_total_amount,
            "Avg_Balance": card_average_balance
        }])

        try:
            card_model = models["card"]

            # Apply the same one-hot encoding used during training
            encoded_input = pd.get_dummies(
                raw_card_input,
                columns=[
                    "sex",
                    "Account_type",
                    "frequency",
                    "type"
                ],
                drop_first=True
            )

            encoded_input = align_features(
                encoded_input,
                card_model
            )

            prediction = int(card_model.predict(encoded_input)[0])

            if hasattr(card_model, "predict_proba"):
                probability = float(
                    card_model.predict_proba(encoded_input)[0][1]
                )
            else:
                probability = float(prediction)

            if probability >= 0.70:
                priority = "High-Priority Prospect"
                recommendation = (
                    "Offer a personalised premium or rewards card."
                )
                result_class = "result-low"

            elif probability >= 0.40:
                priority = "Medium-Priority Prospect"
                recommendation = (
                    "Use a targeted campaign with cashback or joining benefits."
                )
                result_class = "result-info"

            else:
                priority = "Low-Priority Prospect"
                recommendation = (
                    "Focus on customer engagement before making a card offer."
                )
                result_class = "result-high"

            st.markdown(
                f"""
                <div class="{result_class}">
                    <strong>{priority}</strong><br>
                    Card-adoption probability: {probability:.1%}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.info(f"Recommendation: {recommendation}")

        except Exception as error:
            st.error(
                "Prediction failed because the app input features do not "
                f"match the saved card model. Details: {error}"
            )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.caption(
    "Banking Prediction System | K-Means and Random Forest Models"
)