## House Prices - Advanced Regression Techniques

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

### Kaggle Competition Link :
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## ⚠️ Important Note About Models

> **`xgb_model_30.pkl` is created specifically for the Streamlit application.**

This model was trained using only **30 selected features** to simplify user input and improve deployment usability.

The full-feature model (trained using all available features) is stored separately inside the **`models/xgb_model.pkl` folder** and is intended for experimentation, comparison, and advanced evaluation.

Same for **`model_columns.pkl`.

---

### 📌 Model Usage Clarification

- `model/xgb_model_30.pkl` → Used for Streamlit App (UI-friendly version)
- `models/xgb_model.pkl` → full-feature training models (research & development version)

⚡ Please ensure you load the correct model depending on your use case.

### Clone or Download the Project
```bash
git clone https://github.com/pbhttai/House-Price-Predication.git
cd House-Price-Predication
```
