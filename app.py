import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load model and saved columns
# -------------------------------

# this is will predcit only for 30 features

model = joblib.load("models/xgb_model_30.pkl")
model_columns = joblib.load("models/model_columns_30.pkl")

st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏠 House Price Prediction App")
st.write("Fill in the house details below:")

# -------------------------------
# NUMERICAL INPUTS
# -------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    GrLivArea = st.number_input("Ground Living Area (sq ft)", 300, 5000, 1500)
    GarageCars = st.slider("Garage Cars Capacity", 0, 4, 1)
    GarageArea = st.number_input("Garage Area (sq ft)", 0, 1500, 400)
    TotalBsmtSF = st.number_input("Total Basement Area", 0, 3000, 800)
    FirstFlrSF = st.number_input("1st Floor Area", 300, 3000, 900)

with col2:
    SecondFlrSF = st.number_input("2nd Floor Area", 0, 2000, 400)
    FullBath = st.slider("Full Bathrooms", 0, 4, 2)
    HalfBath = st.slider("Half Bathrooms", 0, 2, 1)
    BedroomAbvGr = st.slider("Bedrooms Above Ground", 0, 6, 3)
    KitchenAbvGr = st.slider("Kitchens Above Ground", 0, 3, 1)
    TotRmsAbvGrd = st.slider("Total Rooms Above Ground", 2, 12, 6)

with col3:
    Fireplaces = st.slider("Fireplaces", 0, 3, 1)
    LotArea = st.number_input("Lot Area", 1000, 50000, 8000)
    WoodDeckSF = st.number_input("Wood Deck Area", 0, 1000, 100)
    OpenPorchSF = st.number_input("Open Porch Area", 0, 1000, 50)
    YearBuilt = st.number_input("Year Built", 1900, 2023, 2000)
    YearRemodAdd = st.number_input("Year Remodeled", 1900, 2023, 2005)

GarageYrBlt = st.number_input("Garage Year Built", 1900, 2023, 2000)
YrSold = st.number_input("Year Sold", 2006, 2010, 2010)

# -------------------------------
# CATEGORICAL INPUTS
# -------------------------------

st.subheader("Categorical Features")

col4, col5, col6 = st.columns(3)

with col4:
    MSZoning = st.selectbox("MS Zoning", ["RL", "RM", "FV", "RH", "C (all)"])
    Neighborhood = st.selectbox("Neighborhood", [
        "NAmes","CollgCr","OldTown","Edwards","Somerst",
        "NridgHt","Gilbert","Sawyer","NWAmes","BrkSide"
    ])
    BldgType = st.selectbox("Building Type", ["1Fam","2fmCon","Duplex","TwnhsE","Twnhs"])

with col5:
    HouseStyle = st.selectbox("House Style", ["1Story","2Story","1.5Fin","SLvl","SFoyer"])
    RoofStyle = st.selectbox("Roof Style", ["Gable","Hip","Flat","Gambrel"])
    Exterior1st = st.selectbox("Exterior Material", ["VinylSd","MetalSd","Wd Sdng","HdBoard","BrkFace"])

with col6:
    ExterQual = st.selectbox("Exterior Quality", ["Ex","Gd","TA","Fa"])
    KitchenQual = st.selectbox("Kitchen Quality", ["Ex","Gd","TA","Fa"])
    HeatingQC = st.selectbox("Heating Quality", ["Ex","Gd","TA","Fa"])
    Foundation = st.selectbox("Foundation Type", ["PConc","CBlock","BrkTil","Wood"])
    GarageType = st.selectbox("Garage Type", ["Attchd","Detchd","BuiltIn","CarPort","None"])
    SaleCondition = st.selectbox("Sale Condition", ["Normal","Abnorml","Partial","AdjLand"])

# -------------------------------
# Create DataFrame
# -------------------------------

input_data = {
    'OverallQual': OverallQual,
    'GrLivArea': GrLivArea,
    'GarageCars': GarageCars,
    'GarageArea': GarageArea,
    'TotalBsmtSF': TotalBsmtSF,
    '1stFlrSF': FirstFlrSF,
    '2ndFlrSF': SecondFlrSF,
    'FullBath': FullBath,
    'HalfBath': HalfBath,
    'BedroomAbvGr': BedroomAbvGr,
    'KitchenAbvGr': KitchenAbvGr,
    'TotRmsAbvGrd': TotRmsAbvGrd,
    'Fireplaces': Fireplaces,
    'LotArea': LotArea,
    'WoodDeckSF': WoodDeckSF,
    'OpenPorchSF': OpenPorchSF,
    'MSZoning': MSZoning,
    'Neighborhood': Neighborhood,
    'BldgType': BldgType,
    'HouseStyle': HouseStyle,
    'RoofStyle': RoofStyle,
    'Exterior1st': Exterior1st,
    'ExterQual': ExterQual,
    'KitchenQual': KitchenQual,
    'HeatingQC': HeatingQC,
    'Foundation': Foundation,
    'GarageType': GarageType,
    'SaleCondition': SaleCondition,
}

input_df = pd.DataFrame([input_data])

# -------------------------------
# Feature Engineering
# -------------------------------

input_df["TotalBath"] = input_df["FullBath"] + 0.5 * input_df["HalfBath"]
input_df["HouseAge"] = YrSold - YearBuilt
input_df["RemodelAge"] = YrSold - YearRemodAdd
input_df["GarageAge"] = YrSold - GarageYrBlt

# -------------------------------
# Encoding
# -------------------------------

input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict House Price"):

    prediction_log = model.predict(input_df)
    prediction = np.expm1(prediction_log)

    st.success(f"🏷 Estimated House Price: ₹ {prediction[0]:,.2f}")