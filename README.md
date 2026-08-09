# 🏠 Indian House Price Prediction

An end-to-end Machine Learning project that predicts residential property prices in India using a Random Forest Regression model and an interactive Streamlit web application.

## 🚀 Live Demo

[Open the Streamlit App](YOUR_STREAMLIT_LINK)

## 📌 Project Overview

This project uses an Indian real estate dataset to build a house price prediction system.

The complete workflow includes:

- Data cleaning and preprocessing
- Feature engineering
- Outlier detection and removal
- Categorical feature encoding
- Random Forest Regression
- Hyperparameter tuning using GridSearchCV
- Model evaluation
- Interactive Streamlit deployment
- Feature-importance visualization
- Batch prediction using CSV files

## 🤖 Machine Learning Model

The final model is a **Random Forest Regressor** optimized using GridSearchCV.

### Best Parameters

- `n_estimators`: 200
- `max_depth`: 20
- `min_samples_split`: 5
- `min_samples_leaf`: 1

## 📊 Model Performance

The model achieved:

- **R² Score:** 0.855
- **MAE:** 0.0077
- **RMSE:** 0.0156

## 🖥️ Application Features

The Streamlit application allows users to enter property details such as:

- Carpet Area
- Bathrooms
- Balconies
- Current Floor
- Total Floors
- Location
- Transaction Type
- Furnishing
- Facing
- Ownership

The application then provides an estimated house price.

It also provides a **feature-importance visualization** to help understand which property features contributed most to the prediction.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- GridSearchCV
- Joblib
- Streamlit

## 📂 Project Structure

```text
indian-house-price-prediction/
│
├── app.py
├── feature_names.pkl
├── requirements.txt
└── README.md
