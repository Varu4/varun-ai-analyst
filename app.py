# ==================================================
# VARUN AI DATA ANALYST - SAAS MASTER SYSTEM
# Author: Varun Walekar
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import datetime
import requests
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet


# ================= CONFIG =================

st.set_page_config(
    page_title="Varun AI Analyst",
    page_icon="📊",
    layout="wide"
)


# ================= PASSWORDS =================

STANDARD_PASS = st.secrets["STANDARD_PASS"]
PREMIUM_PASS = st.secrets["PREMIUM_PASS"]



# ================= SESSION =================

if "plan" not in st.session_state:
    st.session_state.plan = "Basic"

if "data" not in st.session_state:
    st.session_state.data = None

if "multi_data" not in st.session_state:
    st.session_state.multi_data = []


# ================= SIDEBAR =================

st.sidebar.title("📊 Varun AI Analyst")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📁 Upload",
        "📈 Analysis",
        "📊 Advanced EDA",
        "⚙ Feature Engineering",
        "📌 Custom KPIs",
        "📄 Report",
        "⬇ Export",
        "💼 Business Intel",
        "🤖 ML Studio",
        "📄 Reports",
        "🌐 API Demo",
        "⏰ Scheduler",
        "💳 Upgrade",
        "👤 Account"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write(f"💼 Plan: **{st.session_state.plan}**")


# ================= UTILITIES =================

def ai_summary(df):

    return f"""
Rows: {df.shape[0]}
Columns: {df.shape[1]}
Missing Values: {df.isnull().sum().sum()}
Numeric Columns: {df.select_dtypes(np.number).shape[1]}
Generated: {datetime.datetime.now()}
"""


def business_insights(df):

    insights = []

    nums = df.select_dtypes(np.number)

    if not nums.empty:

        insights.append(f"Top KPI: {nums.mean().idxmax()}")
        insights.append(f"Weak KPI: {nums.mean().idxmin()}")
        insights.append(f"High Risk: {nums.std().idxmax()}")

    if df.isnull().sum().sum() > 0:
        insights.append("Improve data quality")

    insights.append("Enable automation")

    return insights


def generate_pdf(df, summary):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("AI Business Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Summary", styles["Heading2"]))
    elements.append(Paragraph(summary, styles["Normal"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Insights", styles["Heading2"]))

    for i in business_insights(df):
        elements.append(Paragraph("- " + i, styles["Normal"]))

    doc.build(elements)

    buffer.seek(0)

    return buffer


# ================= HOME =================

if menu == "🏠 Home":

    st.title("🤖 AI Data Analyst Platform")

    st.subheader("AI-Powered Analytics for Business & Freelancing")

    st.markdown("""
### Analyze • Predict • Report • Grow

✔ Data Cleaning  
✔ AI Reports  
✔ Machine Learning  
✔ Business Intelligence  
✔ Automation  

✅ Free Demo (Basic Access)
""")

    st.success("Upload your dataset to start 🚀")


# ================= UPLOAD =================

elif menu == "📁 Upload":

    st.title("📁 Upload Dataset")

    st.info("Basic: 2MB | Standard/Premium: Unlimited")

    files = st.file_uploader(
        "Upload CSV / Excel",
        ["csv", "xlsx"],
        accept_multiple_files=True
    )

    if files:

        for file in files:

            if st.session_state.plan == "Basic" and file.size > 2*1024*1024:
                st.error("Basic limit: 2MB")
                continue

            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            st.session_state.data = df
            st.session_state.multi_data.append(df)

        st.success("Uploaded Successfully")

        st.dataframe(st.session_state.data.head())

# ================= ANALYSIS =================

elif menu == "📈 Analysis":

    if st.session_state.data is None:
        st.warning("Upload data first")
        st.stop()

    df = st.session_state.data

    st.title("📈 Data Overview & Analysis")

    # -------- BASIC FEATURES --------
    st.subheader("📄 Dataset Preview")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Head",
        "Tail",
        "Shape",
        "Columns",
        "Dtypes",
        "Info"
    ])

    with tab1:
        st.write("First 5 Rows")
        st.dataframe(df.head())

    with tab2:
        st.write("Last 5 Rows")
        st.dataframe(df.tail())

    with tab3:
        st.write("Dataset Shape")
        st.info(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    with tab4:
        st.write("Column Names")
        st.write(list(df.columns))

    with tab5:
        st.write("Data Types")
        st.dataframe(df.dtypes)

    with tab6:
        st.write("Dataset Info")

        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()

        st.text(s)

    st.markdown("---")

    # -------- FULL DATA VIEW --------
    st.subheader("📊 Full Dataset")

    if st.checkbox("Show Full Data"):
        st.dataframe(df)

    st.markdown("---")

    # -------- CHART (STANDARD+) --------
    if st.session_state.plan == "Basic":
        st.warning("🔒 Upgrade to Standard to use Charts")
        st.stop()

    st.subheader("📈 Visualization")

    num_cols = df.select_dtypes(np.number).columns

    if len(num_cols) == 0:
        st.warning("No numeric columns found")
        st.stop()

    col = st.selectbox("Select Column", num_cols)

    chart = st.selectbox(
        "Chart Type",
        ["Line", "Bar", "Histogram", "Pie"]
    )

    fig, ax = plt.subplots()

    if chart == "Line":
        df[col].plot(ax=ax)

    elif chart == "Bar":
        df[col].value_counts().plot.bar(ax=ax)

    elif chart == "Histogram":
        df[col].plot.hist(ax=ax)

    elif chart == "Pie":
        df[col].value_counts().plot.pie(ax=ax)

    st.pyplot(fig)


# ================= ADVANCED EDA =================

elif menu == "📊 Advanced EDA":

    if st.session_state.plan == "Basic":
        st.error("🔒 Upgrade Required")
        st.stop()

    if st.session_state.data is None:
        st.warning("Upload data first")
        st.stop()

    st.title("📊 Advanced Exploratory Data Analysis")

    df = st.session_state.data

    numeric_cols = df.select_dtypes(np.number).columns

    # ---------- STATISTICAL SUMMARY ----------
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe())

    st.markdown("---")

    # ---------- MISSING VALUES ----------
    st.subheader("❓ Missing Values Analysis")

    missing = df.isnull().sum()

    st.dataframe(missing)

    if missing.sum() > 0:

        st.warning("Missing values detected")

        fill_option = st.selectbox(
            "Handle Missing Values",
            ["Do Nothing", "Fill with Mean", "Fill with Median", "Fill with Mode", "Drop Rows"]
        )

        if st.button("Apply Missing Value Treatment"):

            if fill_option == "Fill with Mean":
                df.fillna(df.mean(numeric_only=True), inplace=True)

            elif fill_option == "Fill with Median":
                df.fillna(df.median(numeric_only=True), inplace=True)

            elif fill_option == "Fill with Mode":
                df.fillna(df.mode().iloc[0], inplace=True)

            elif fill_option == "Drop Rows":
                df.dropna(inplace=True)

            st.session_state.data = df
            st.success("Missing Values Handled ✔")

    st.markdown("---")

    # ---------- CORRELATION ----------
    if len(numeric_cols) > 1:

        st.subheader("🔗 Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.heatmap(
            df[numeric_cols].corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)

    st.markdown("---")

    # ---------- OUTLIER DETECTION ----------
    st.subheader("📦 Outlier Detection (Boxplot)")

    if len(numeric_cols) == 0:
        st.warning("No numeric columns found")
        st.stop()

    selected_col = st.selectbox(
        "Select Column for Outlier Analysis",
        numeric_cols
    )

    fig2, ax2 = plt.subplots()

    sns.boxplot(x=df[selected_col], ax=ax2)

    st.pyplot(fig2)

    # ---------- OUTLIER HANDLING ----------
    st.subheader("🛠 Handle Outliers (IQR Method)")

    Q1 = df[selected_col].quantile(0.25)
    Q3 = df[selected_col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    st.info(f"Lower Bound: {lower:.2f} | Upper Bound: {upper:.2f}")

    out_method = st.selectbox(
        "Outlier Treatment Method",
        ["Do Nothing", "Remove Outliers", "Cap Outliers"]
    )

    if st.button("Apply Outlier Treatment"):

        if out_method == "Remove Outliers":

            df = df[(df[selected_col] >= lower) & (df[selected_col] <= upper)]

            st.session_state.data = df

            st.success("Outliers Removed ✔")

        elif out_method == "Cap Outliers":

            df[selected_col] = np.where(
                df[selected_col] < lower,
                lower,
                np.where(df[selected_col] > upper, upper, df[selected_col])
            )

            st.session_state.data = df

            st.success("Outliers Capped ✔")



# ================= FEATURE ENGINEERING =================

elif menu == "⚙ Feature Engineering":

    if st.session_state.plan == "Basic":
        st.error("🔒 Upgrade Required")
        st.stop()

    st.title("⚙ Feature Engineering")

    df = st.session_state.data

    if df is None:
        st.warning("Upload data first")
        st.stop()

    # ================= COLUMN TYPES =================

    numeric_cols = [col for col in df.columns if df[col].dtype != "object"]
    cat_cols = [col for col in df.columns if df[col].dtype == "object"]

    st.subheader("📌 Column Types")

    st.write("🔢 Numerical Columns:")
    st.write(numeric_cols)

    st.write("🔤 Categorical Columns:")
    st.write(cat_cols)

    st.markdown("---")

    # ================= NUMERIC TRANSFORMATION =================

    st.subheader("📐 Numeric Transformations")

    num_col = st.selectbox("Select Numeric Column", numeric_cols)

    option = st.radio(
        "Transformation",
        ["Square", "Log", "Normalize"]
    )

    if st.button("Apply Transformation"):

        if option == "Square":
            df[num_col + "_sq"] = df[num_col] ** 2

        elif option == "Log":
            df[num_col + "_log"] = np.log1p(df[num_col])

        elif option == "Normalize":
            df[num_col + "_norm"] = (
                (df[num_col] - df[num_col].min()) /
                (df[num_col].max() - df[num_col].min())
            )

        st.session_state.data = df

        st.success("✅ Numeric Feature Created")

    st.markdown("---")

    # ================= ENCODING =================

    st.subheader("🔁 Categorical Encoding")

    enc_col = st.selectbox("Select Categorical Column", cat_cols)

    encode_type = st.radio(
        "Encoding Method",
        ["Label Encoding", "One-Hot Encoding"]
    )

    st.warning("⚠️ Do NOT apply Label + One-Hot on the same column.")

    if st.button("Apply Encoding"):

        from sklearn.preprocessing import LabelEncoder

        # ----- Label Encoding -----
        if encode_type == "Label Encoding":

            le = LabelEncoder()

            df[enc_col + "_label"] = le.fit_transform(df[enc_col])

            st.success("✅ Label Encoding Applied")

        # ----- One Hot Encoding -----
        elif encode_type == "One-Hot Encoding":

            dummies = pd.get_dummies(df[enc_col], prefix=enc_col)

            df = pd.concat([df, dummies], axis=1)

            st.success("✅ One-Hot Encoding Applied")

        st.session_state.data = df

    st.markdown("---")

    # ================= PREVIEW =================

    st.subheader("📊 Updated Dataset Preview")

    st.dataframe(df.head())



# ================= KPI =================

elif menu == "📌 Custom KPIs":

    if st.session_state.plan == "Basic":
        st.error("🔒 Upgrade Required")
        st.stop()

    st.title("📌 Custom KPIs")

    df = st.session_state.data

    num = df.select_dtypes(np.number).columns

    col = st.selectbox("Metric", num)

    kpi = st.selectbox(
        "Type",
        ["Average","Maximum","Minimum","Total"]
    )

    if st.button("Generate"):

        if kpi=="Average":
            val = df[col].mean()
        elif kpi=="Maximum":
            val = df[col].max()
        elif kpi=="Minimum":
            val = df[col].min()
        elif kpi=="Total":
            val = df[col].sum()

        st.metric(kpi, f"{val:.2f}")


# ================= EXPORT =================

elif menu == "⬇ Export":

    if st.session_state.plan == "Basic":
        st.error("🔒 Upgrade Required")
        st.stop()

    st.title("⬇ Export")

    df = st.session_state.data

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "data.csv",
        "text/csv"
    )

# ================= ML =================

elif menu == "🤖 ML Studio":

    if st.session_state.plan != "Premium":
        st.error("🔒 Premium Only")
        st.stop()

    st.title("🤖 ML Studio - Machine Learning Lab")

    df = st.session_state.data

    if df is None:
        st.warning("Upload data first")
        st.stop()

    # ================= TARGET & FEATURES =================

    st.subheader("📌 Select Target & Features")

    target = st.selectbox("Select Target (Dependent Variable)", df.columns)

    feature_cols = st.multiselect(
        "Select Features (Independent Variables)",
        df.select_dtypes(np.number).columns
    )

    test_size = st.slider(
        "Test Size (%)",
        min_value=10,
        max_value=50,
        value=20
    ) / 100

    st.markdown("---")

    # ================= MODEL SELECTION =================

    st.subheader("⚙ Select Model")

    model_type = st.selectbox(
        "Choose Algorithm",
        [
            "Linear Regression",
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting"
        ]
    )

    st.markdown("---")

    # ================= TRAIN =================

    if st.button("🚀 Train Model"):

        # ---------------- Split X and y ----------------

        X = df[feature_cols]
        y = df[target]

        # ---------------- Train Test Split ----------------

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42
        )

        # ---------------- Import Models ----------------

        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            classification_report,
            mean_squared_error,
            r2_score
        )

        # ---------------- Detect Problem Type ----------------

        is_classification = y.nunique() <= 10

        # ---------------- Build Model ----------------

        if model_type == "Linear Regression":
            model = LinearRegression()

        elif model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)

        elif model_type == "Decision Tree":

            if is_classification:
                model = DecisionTreeClassifier()
            else:
                model = DecisionTreeRegressor()

        elif model_type == "Random Forest":

            if is_classification:
                model = RandomForestClassifier()
            else:
                model = RandomForestRegressor()

        elif model_type == "Gradient Boosting":

            if is_classification:
                model = GradientBoostingClassifier()
            else:
                model = GradientBoostingRegressor()

        # ---------------- Train ----------------

        model.fit(X_train, y_train)

        # ---------------- Predict ----------------

        y_pred = model.predict(X_test)

        st.success("✅ Model Trained Successfully")

        st.markdown("---")

        # ================= RESULTS =================

        st.subheader("📊 Model Performance")

        # -------- Classification --------
        if is_classification:

            acc = accuracy_score(y_test, y_pred)

            cm = confusion_matrix(y_test, y_pred)

            st.metric("Accuracy", f"{acc:.2f}")

            st.write("### Confusion Matrix")
            st.dataframe(cm)

            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))


        # -------- Regression --------
        else:

            mse = mean_squared_error(y_test, y_pred)

            r2 = r2_score(y_test, y_pred)

            st.metric("MSE", f"{mse:.2f}")
            st.metric("R² Score", f"{r2:.2f}")

        st.markdown("---")

        # ================= SAMPLE PREDICTIONS =================

        st.subheader("🔍 Sample Predictions")

        result_df = pd.DataFrame({
            "Actual": y_test.values[:10],
            "Predicted": y_pred[:10]
        })

        st.dataframe(result_df)

        # ================= SAVE MODEL =================

        joblib.dump(model, "trained_model.pkl")

        st.success("💾 Model Saved as trained_model.pkl")



# ================= REPORT =================

elif menu == "📄 Reports":

    if st.session_state.data is None:
        st.stop()

    st.title("📄 Reports")

    df = st.session_state.data

    summary = ai_summary(df)

    if st.button("Generate PDF"):

        pdf = generate_pdf(df, summary)

        st.download_button(
            "Download",
            pdf,
            "report.pdf",
            "application/pdf"
        )


# ================= UPGRADE =================

elif menu == "💳 Upgrade":

    st.title("💳 Upgrade Your Plan")

    st.markdown("""
### Unlock Advanced Features

✔ Full Reports  
✔ ML Models  
✔ Business Insights  
✔ Unlimited Upload  
✔ Priority Support
""")

    st.info("💬 Pay on Fiverr → Get Password → Unlock Access")

    st.markdown("👉 https://www.fiverr.com/varunwalekar04")

    st.markdown("---")

    st.subheader("🔓 Enter Access Password")

    col1, col2, col3 = st.columns(3)

    # -------- BASIC --------
    with col1:
        st.subheader("Basic")
        st.write("Free")
        st.write("✔ Limited Access")

    # -------- STANDARD --------
    with col2:

        st.subheader("Standard")

        pwd = st.text_input("Standard Password", type="password")

        if st.button("Unlock Standard"):

            if pwd == STANDARD_PASS:

                st.session_state.plan = "Standard"
                st.success("✅ Standard Activated")

            else:
                st.error("❌ Wrong Password")

    # -------- PREMIUM --------
    with col3:

        st.subheader("Premium")

        pwd2 = st.text_input("Premium Password", type="password")

        if st.button("Unlock Premium"):

            if pwd2 == PREMIUM_PASS:

                st.session_state.plan = "Premium"
                st.success("🚀 Premium Activated")

            else:
                st.error("❌ Wrong Password")


# ================= ACCOUNT =================

elif menu == "👤 Account":

    st.title("👤 Account")

    st.write(f"Plan: {st.session_state.plan}")

    st.write("📧 yourname@gmail.com")
    st.write("💼 Fiverr: fiverr.com/varunwalekar04")

    msg = st.text_area("Message")

    if st.button("Send"):
        st.success("Message Sent ✔")






