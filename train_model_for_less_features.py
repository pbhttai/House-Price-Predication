# ============================================
# 1️⃣ Import Libraries
# ============================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


# ============================================
# 2️⃣ Load Dataset
# ============================================

df = pd.read_csv("notebooks/train.csv")   


# ============================================
# 3️⃣ Feature Engineering (Before Selection)
# ============================================

# Age features
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
df["RemodelAge"] = df["YrSold"] - df["YearRemodAdd"]
df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]

# Bathroom feature
df["TotalBath"] = (
    df["FullBath"]
    + 0.5 * df["HalfBath"]
    + df["BsmtFullBath"]
    + 0.5 * df["BsmtHalfBath"]
)

# Fill GarageAge missing (if no garage)
df["GarageAge"] = df["GarageAge"].fillna(0)


# ============================================
# 4️⃣ Select Only 30 Main Features
# ============================================

selected_features = [
    # Numerical
    'OverallQual','GrLivArea','GarageCars','GarageArea',
    'TotalBsmtSF','1stFlrSF','2ndFlrSF','FullBath','HalfBath',
    'BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces',
    'LotArea','WoodDeckSF','OpenPorchSF',
    'HouseAge','RemodelAge','GarageAge','TotalBath',

    # Categorical
    'MSZoning','Neighborhood','BldgType','HouseStyle','RoofStyle',
    'Exterior1st','ExterQual','KitchenQual','HeatingQC',
    'Foundation','GarageType','SaleCondition'
]

df_small = df[selected_features + ["SalePrice"]]


# ============================================
# 5️⃣ Log Transform Target
# ============================================

df_small["SalePrice"] = np.log1p(df_small["SalePrice"])


# ============================================
# 6️⃣ Split Features and Target
# ============================================

X = df_small.drop("SalePrice", axis=1)
y = df_small["SalePrice"]


# ============================================
# 7️⃣ One-Hot Encoding
# ============================================

X = pd.get_dummies(X, drop_first=True)


# ============================================
# 8️⃣ Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ============================================
# 9️⃣ Train XGBoost Model
# ============================================

model = XGBRegressor(
    n_estimators=900,
    learning_rate=0.1,
    max_depth=2,
    min_child_weight=5,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0.2,
    reg_alpha=2,
    reg_lambda=5,
    random_state=42,
    n_jobs=-1,
    booster='gbtree',
    base_score= 0.25,
)

model.fit(X_train, y_train)


# ============================================
# 🔟 Evaluation
# ============================================

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print("Train RMSE (log scale):", train_rmse)
print("Test RMSE (log scale):", test_rmse)


# Convert back to original price scale
test_pred_original = np.expm1(test_pred)
y_test_original = np.expm1(y_test)

rmse_original = np.sqrt(mean_squared_error(y_test_original, test_pred_original))

print("Test RMSE (original price):", rmse_original)


# ============================================
# 1️⃣1️⃣ Save Model + Columns
# ============================================

joblib.dump(model, "models/xgb_model_30.pkl")
joblib.dump(X.columns.tolist(), "models/model_columns_30.pkl")

print("Model and columns saved successfully!")