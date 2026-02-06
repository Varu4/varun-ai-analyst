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

    st.title("📊 Advanced EDA")

    df = st.session_state.data

    numeric_cols = df.select_dtypes(np.number).columns

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    if len(numeric_cols) > 1:

        fig, ax = plt.subplots(figsize=(8,5))

        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)

        st.pyplot(fig)


# ================= FEATURE ENGINEERING =================

elif menu == "⚙ Feature Engineering":

    if st.session_state.plan == "Basic":
        st.error("🔒 Upgrade Required")
        st.stop()

    st.title("⚙ Feature Engineering")

    df = st.session_state.data

    num = df.select_dtypes(np.number).columns

    col = st.selectbox("Select Column", num)

    option = st.radio(
        "Transformation",
        ["Square","Log","Normalize"]
    )

    if st.button("Apply"):

        if option=="Square":
            df[col+"_sq"] = df[col]**2

        elif option=="Log":
            df[col+"_log"] = np.log1p(df[col])

        elif option=="Normalize":
            df[col+"_norm"] = (df[col]-df[col].min())/(df[col].max()-df[col].min())

        st.session_state.data = df

        st.success("Feature Created ✔")


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
        st.error("Premium Only")
        st.stop()

    df = st.session_state.data

    st.title("🤖 ML Studio")

    target = st.selectbox("Target", df.columns)

    features = st.multiselect(
        "Features",
        df.select_dtypes(np.number).columns
    )

    if st.button("Train"):

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X,y)

        model = LinearRegression()

        model.fit(X_train,y_train)

        score = model.score(X_test,y_test)

        st.success(f"Model Score: {score:.2f}")


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



