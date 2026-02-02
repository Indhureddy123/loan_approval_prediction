import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Loan Approval Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["Overview", "EDA", "Model Metrics", "Prediction"]
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("loan_approval_dataset (1).csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ================= OVERVIEW =================
if menu == "Overview":
    st.title("🏦 Smart Loan Approval Prediction Platform")

    st.markdown("""
    ### Project Objective
    This application demonstrates a **complete Machine Learning pipeline**
    designed to predict **loan approval decisions** based on applicant financial
    and credit-related attributes.
    """)

    st.subheader("📊 Dataset Snapshot")
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Records", df.shape[0])
    col2.metric("Number of Features", df.shape[1])
    col3.metric("Total Missing Values", df.isnull().sum().sum())

    st.subheader("🔍 Sample Records")
    st.dataframe(df.head(15))

    st.subheader("📈 Statistical Overview")
    st.dataframe(df.describe())

    st.info(
        "This dataset is used to train classification models that assist financial "
        "institutions in making faster and data-driven loan decisions."
    )

# ================= EDA =================
elif menu == "EDA":
    st.title("📊 Data Exploration & Insights")

    st.subheader("Loan Approval Outcome Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="loan_status", data=df, ax=ax1)
    ax1.set_xlabel("Loan Status")
    ax1.set_ylabel("Applicant Count")
    st.pyplot(fig1)

    st.subheader("Feature Correlation Analysis")

    numeric_df = df.select_dtypes(include=["int64", "float64"])
    numeric_df = numeric_df.drop("loan_status", axis=1, errors="ignore")

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="viridis",
        linewidths=0.5,
        ax=ax2
    )
    st.pyplot(fig2)

    st.subheader("🧠 Observations from EDA")
    st.markdown("""
    - Applicants with **higher CIBIL scores** show a strong likelihood of approval  
    - **Income, assets, and credit history** play a crucial role  
    - Certain financial features are correlated, making **tree-based models effective**  
    - Data patterns justify the use of **classification algorithms**
    """)

# ================= MODEL METRICS =================
elif menu == "Model Metrics":
    st.title("📈 Model Evaluation & Comparison")

    st.markdown("""
    This section presents the **performance evaluation** of the trained models
    using standard classification metrics.
    """)

    metrics = pd.read_csv("model_metrics.csv")
    st.subheader("📋 Model Performance Summary")
    st.dataframe(metrics)

    st.subheader("🔲 Confusion Matrix")
    st.image("confusion_matrix.png", use_column_width=True)

    st.subheader("📉 ROC Curve Analysis")
    st.image("roc_curve.png", use_column_width=True)

    st.success(
        "The selected model demonstrates a good balance between precision and recall, "
        "making it suitable for real-world loan approval systems."
    )

# ================= PREDICTION =================
elif menu == "Prediction":
    st.title("🔮 Loan Approval Simulator")

    st.markdown("""
    Enter applicant details below to **simulate a loan approval decision**
    using the trained Machine Learning model.
    """)

    model = joblib.load("loan_approval_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")

    st.subheader("📝 Applicant Information")

    user_input = []
    for feature in features:
        value = st.number_input(
            label=f"{feature}",
            min_value=0.0,
            value=0.0,
            step=1.0
        )
        user_input.append(value)

    if st.button("Check Loan Eligibility"):
        input_array = np.array(user_input).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.markdown(f"### 📊 Approval Probability: **{probability:.2%}**")

        if prediction == 1:
            st.success("🎉 Congratulations! The loan is likely to be **Approved**.")
        else:
            st.error("⚠️ Unfortunately, the loan is likely to be **Rejected**.")

        st.caption(
            "Note: This prediction is based on historical data and should be used "
            "for educational or decision-support purposes only."
        )
