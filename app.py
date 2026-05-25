# ---------- Import Libraries ----------
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# ---------- Load Model ----------
model = pickle.load(open("linear_model.pkl", "rb"))

# ---------- App Title ----------
st.title("🏠 House Price Prediction App")

st.write(
    """
    This application predicts house prices using
    a trained Linear Regression model.
    """
)

# =========================================
# USER INPUTS
# =========================================

st.header("Enter House Details")

# ---------- Numeric Inputs ----------

Taxi_dist = st.number_input(
    "Taxi Distance",
    min_value=0.0,
    value=10.0
)

Market_dist = st.number_input(
    "Market Distance",
    min_value=0.0,
    value=5.0
)

Hospital_dist = st.number_input(
    "Hospital Distance",
    min_value=0.0,
    value=3.0
)

Carpet_area = st.number_input(
    "Carpet Area",
    min_value=0.0,
    value=1000.0
)

Builtup_area = st.number_input(
    "Builtup Area",
    min_value=0.0,
    value=1200.0
)

Rainfall = st.number_input(
    "Rainfall",
    min_value=0,
    value=100
)

# =========================================
# CATEGORICAL INPUTS
# =========================================

Parking_type = st.selectbox(
    "Parking Type",
    ["Open", "Covered", "No Parking", "Not Provided"]
)

City_type = st.selectbox(
    "City Type",
    ["CAT A", "CAT B", "CAT C"]
)

# =========================================
# MANUAL ONE-HOT ENCODING
# =========================================

# Parking Type Encoding

Parking_type_No_Parking = 0
Parking_type_Not_Provided = 0
Parking_type_Open = 0

if Parking_type == "No Parking":
    Parking_type_No_Parking = 1

elif Parking_type == "Not Provided":
    Parking_type_Not_Provided = 1

elif Parking_type == "Open":
    Parking_type_Open = 1

# CAT A becomes default category because of drop_first=True

# City Type Encoding

City_type_CAT_B = 0
City_type_CAT_C = 0

if City_type == "CAT B":
    City_type_CAT_B = 1

elif City_type == "CAT C":
    City_type_CAT_C = 1

# =========================================
# CREATE FEATURE ARRAY
# =========================================

features = np.array([[
    Taxi_dist,
    Market_dist,
    Hospital_dist,
    Carpet_area,
    Builtup_area,
    Rainfall,
    Parking_type_No_Parking,
    Parking_type_Not_Provided,
    Parking_type_Open,
    City_type_CAT_B,
    City_type_CAT_C
]])

# =========================================
# PREDICTION BUTTON
# =========================================

if st.button("Predict House Price"):

    prediction = model.predict(features)

    st.success(
        f"🏡 Predicted House Price: ₹ {prediction[0]:,.2f}"
    )

# =========================================
# SIDEBAR
# =========================================

st.sidebar.header("About Project")

st.sidebar.write(
    """
    Machine Learning Project:
    
    - Algorithm: Linear Regression
    - Problem Type: Regression
    - Frontend: Streamlit
    - Goal: Predict House Prices
    """
)

# =========================================
# FOOTER
# =========================================

st.markdown("---")

