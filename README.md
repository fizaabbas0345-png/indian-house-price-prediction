---
title: House Price Prediction
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# 🏠 House Price Prediction

A Streamlit app that predicts Indian house prices from property details
(carpet area, bathrooms, balconies, floor, location, etc.) using a
Random Forest model.

## Model details

- Algorithm: Random Forest Regressor (tuned via GridSearchCV)
- Test R²: 0.859
- Test MAE: 0.128 (log scale) — roughly ±13–14% average error in rupee terms
- Trained mainly on properties between **700–1500 sqft** carpet area.
  Predictions outside that range are flagged as less reliable in the app.

## Files

- `app.py` — the Streamlit app
- `house_price_model.pkl` — trained Random Forest model
- `feature_names.pkl` — feature column order expected by the model
- `requirements.txt` — pinned dependencies (scikit-learn pinned to match
  the version the model was trained with)

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
