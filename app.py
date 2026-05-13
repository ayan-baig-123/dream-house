# ==========================================
# DREAM HOUSE AI
# ADVANCED STREAMLIT APP
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Dream House",
    page_icon="🏠",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );
    color: white;
}

h1,h2,h3,h4,h5,h6,p,label {
    color: white !important;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(to right,#06b6d4,#3b82f6);
    color: white;
    border: none;
    border-radius: 15px;
    height: 3.5em;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover{
    transform: scale(1.03);
}

.block-container{
    padding-top: 2rem;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 20px rgba(255,255,255,0.05);
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.markdown("""
<h1 style='text-align:center;font-size:55px;'>
🏠 Dream House AI
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h3 style='text-align:center;color:#cbd5e1;'>
Luxury House Price Prediction System
</h3>
""", unsafe_allow_html=True)

st.write("")

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("data.csv")

# ==========================================
# CLEANING
# ==========================================

df.drop_duplicates(inplace=True)

df = df[df["price"] > 0]

# ==========================================
# FEATURE ENGINEERING
# ==========================================

df["house_age"] = 2026 - df["yr_built"]

df["renovated"] = np.where(
    df["yr_renovated"] > 0,
    1,
    0
)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Visual Analytics",
        "AI Prediction"
    ]
)

# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    st.subheader("📊 Housing Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class='card'>
        <h1>{df.shape[0]}</h1>
        <h4>Total Houses</h4>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='card'>
        <h1>{df.shape[1]}</h1>
        <h4>Total Features</h4>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_price = int(df["price"].mean())

        st.markdown(f"""
        <div class='card'>
        <h1>${avg_price}</h1>
        <h4>Average Price</h4>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.subheader("📄 Dataset Preview")

    st.dataframe(df.head())

# ==========================================
# VISUAL ANALYTICS
# ==========================================

elif page == "Visual Analytics":

    st.subheader("📈 Visual Analytics")

    # PRICE DISTRIBUTION

    fig, ax = plt.subplots(figsize=(10,5))

    sns.histplot(
        df["price"],
        bins=50,
        kde=True,
        ax=ax
    )

    ax.set_title("Price Distribution")

    st.pyplot(fig)

    # SQFT VS PRICE

    fig, ax = plt.subplots(figsize=(10,5))

    sns.scatterplot(
        x=df["sqft_living"],
        y=df["price"],
        ax=ax
    )

    ax.set_title("Sqft Living vs Price")

    st.pyplot(fig)

    # HEATMAP

    st.subheader("🔥 Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(14,8))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    st.pyplot(fig)

# ==========================================
# AI PREDICTION
# ==========================================

elif page == "AI Prediction":

    st.subheader("🤖 AI House Price Prediction")

    # MODEL

    x = np.array([
        df["bathrooms"],
        df["bedrooms"],
        df["sqft_living"],
        df["sqft_lot"],
        df["floors"],
        df["waterfront"],
        df["view"],
        df["house_age"]
    ]).T

    y = np.array(df["price"])

    model = LinearRegression()

    model.fit(x, y)

    # USER INPUTS

    col1, col2 = st.columns(2)

    with col1:

        bathrooms = st.slider(
            "Bathrooms",
            1,
            10,
            2
        )

        bedrooms = st.slider(
            "Bedrooms",
            1,
            10,
            3
        )

        sqft_living = st.number_input(
            "Sqft Living",
            500,
            10000,
            2000
        )

        sqft_lot = st.number_input(
            "Sqft Lot",
            500,
            50000,
            5000
        )

    with col2:

        floors = st.slider(
            "Floors",
            1,
            5,
            1
        )

        waterfront = st.selectbox(
            "Waterfront",
            [0,1]
        )

        view = st.slider(
            "View Rating",
            0,
            4,
            2
        )

        house_age = st.slider(
            "House Age",
            1,
            120,
            20
        )

    # PREDICT BUTTON

    if st.button("🚀 Predict Price"):

        values = np.array([[
            bathrooms,
            bedrooms,
            sqft_living,
            sqft_lot,
            floors,
            waterfront,
            view,
            house_age
        ]])

        prediction = model.predict(values)

        st.success(
            f"🏠 Estimated House Price: ${prediction[0]:,.2f}"
        )

        st.balloons()

# ==========================================
# FOOTER
# ==========================================

st.write("")
st.markdown("""
<hr>
<center>
<h4>Made by Ayan Baig</h4>
</center>
""", unsafe_allow_html=True)