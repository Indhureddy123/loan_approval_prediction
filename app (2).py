import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ---------------- SIDEBAR MENU ----------------
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["Overview", "EDA", "Model Metrics", "Prediction"]
)


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("loan_approval_dataset.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ---------------- OVERVIEW PAGE ----------------
if menu == "Overview":
    st.title("🏦 Loan Approval Prediction System")

    st.markdown("""
    ### End-to-End Machine Learning Project
    This application predicts whether a loan will be **Approved** or **Rejected**
    using Machine Learning classification models.
    """)

    st.subheader("📌 Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(20))

    st.subheader("📊 Summary Statistics")
    st.dataframe(df.describe())


# ---------------- EDA PAGE ----------------
elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Loan Status Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="loan_status", data=df, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Correlation Heatmap")

    # Select numeric columns only
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    numeric_df = numeric_df.drop("loan_status", axis=1, errors="ignore")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.subheader("📌 Key Insights")
    st.markdown("""
    - Higher **CIBIL score** increases approval probability  
    - Higher **income & assets** lead to better approval chances  
    - Asset features show correlation → tree models are suitable  
    """)


# ---------------- MODEL METRICS PAGE ----------------
elif menu == "Model Metrics":
    st.title("📈 Model Performance & Evaluation")

    metrics = pd.read_csv("model_metrics.csv")
    st.subheader("Model Comparison")
    st.dataframe(metrics)

    st.subheader("Confusion Matrix")
    st.image("confusion_matrix.png", use_column_width=True)

    st.subheader("ROC Curve")
    st.image("roc_curve.png", use_column_width=True)
# ---------------- PREDICTION PAGE ----------------
elif menu == "Prediction":
    st.title("🔮 Loan Approval Prediction")

    model = joblib.load("loan_approval_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")

    st.subheader("Enter Applicant Details")

    user_input = []
    for feature in features:
        value = st.number_input(feature, value=0.0)
        user_input.append(value)

    if st.button("Predict Loan Status"):
        input_array = np.array(user_input).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.write(f"### Approval Probability: {probability:.2f}")

        if prediction == 1:
            st.success("Loan Approved ✅")
        else:
            st.error("Loan Rejected ❌")
