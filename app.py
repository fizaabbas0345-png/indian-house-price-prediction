import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------
# Load Model and Feature Names (cached so it only loads once)
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("house_price_model.pkl")
        feature_names = joblib.load("feature_names.pkl")
        return model, feature_names
    except FileNotFoundError as e:
        st.error(
            "Model files not found. Make sure 'house_price_model.pkl' and "
            "'feature_names.pkl' are in the same directory as this app."
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()


model, feature_names = load_model()

st.title("🏠 House Price Prediction")
st.write("Enter the house details below to predict the price.")

# -----------------------------
# Reset support
# -----------------------------
if st.button("🔄 Reset form"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# -----------------------------
# Numerical Inputs
# -----------------------------
st.subheader("Property Details")
col1, col2 = st.columns(2)

with col1:
    carpet_area = st.number_input(
        "Carpet Area (sqft)", min_value=100, max_value=10000,
        value=1000, key="carpet_area"
    )
    bathroom = st.number_input(
        "Bathrooms", min_value=1, max_value=10, value=2, key="bathroom"
    )
    balcony = st.number_input(
        "Balconies", min_value=0, max_value=10, value=1, key="balcony"
    )

with col2:
    current_floor = st.number_input(
        "Current Floor", min_value=0, max_value=100, value=1, key="current_floor"
    )
    total_floors = st.number_input(
        "Total Floors", min_value=1, max_value=100, value=5, key="total_floors"
    )

# Validate floor logic immediately, before prediction
floor_error = current_floor > total_floors
if floor_error:
    st.warning("⚠️ Current floor can't be greater than total floors.")

# -----------------------------
# Categorical Inputs
# -----------------------------
st.subheader("Other Details")
col3, col4 = st.columns(2)

with col3:
    location = st.selectbox(
        "Location",
        [
            "ahmadnagar", "ahmedabad", "allahabad", "aurangabad", "badlapur",
            "bangalore", "belgaum", "bhiwadi", "bhiwandi", "bhopal",
            "bhubaneswar", "chandigarh", "chennai", "coimbatore", "dehradun",
            "durgapur", "ernakulam", "faridabad", "ghaziabad", "goa",
            "greater-noida", "guntur", "gurgaon", "guwahati", "gwalior",
            "haridwar", "hyderabad", "indore", "jabalpur", "jaipur",
            "jamshedpur", "jodhpur", "kalyan", "kanpur", "kochi",
            "kolkata", "kozhikode", "lucknow", "ludhiana", "madurai",
            "mangalore", "mohali", "mumbai", "mysore", "nagpur",
            "nashik", "navi-mumbai", "navsari", "nellore", "new-delhi",
            "noida", "palakkad", "palghar", "panchkula", "patna",
            "pondicherry", "pune", "raipur", "rajahmundry", "ranchi",
            "satara", "shimla", "siliguri", "solapur", "sonipat",
            "surat", "thane", "thrissur", "tirupati", "trichy",
            "trivandrum", "udaipur", "udupi", "vadodara", "vapi",
            "varanasi", "vijayawada", "visakhapatnam", "vrindavan", "zirakpur"
        ],
        key="location"
    )

    transaction = st.selectbox(
        "Transaction", ["Other", "Rent/Lease", "Resale"], key="transaction"
    )

    furnishing = st.selectbox(
        "Furnishing", ["Semi-Furnished", "Unfurnished"], key="furnishing"
    )

with col4:
    facing = st.selectbox(
        "Facing",
        [
            "North", "North - East", "North - West", "South",
            "South - East", "South -West", "Unknown", "West"
        ],
        key="facing"
    )

    ownership = st.selectbox(
        "Ownership",
        ["Freehold", "Leasehold", "Power Of Attorney", "Unknown"],
        key="ownership"
    )

# -----------------------------
# Helper: build model input row
# -----------------------------
def build_input_row(carpet_area, bathroom, balcony, current_floor, total_floors,
                     location, transaction, furnishing, facing, ownership):
    input_data = {feature: 0 for feature in feature_names}

    input_data["Carpet Area"] = carpet_area
    input_data["Bathroom"] = bathroom
    input_data["Balcony"] = balcony
    input_data["Current_Floor"] = current_floor
    input_data["Total_Floors"] = total_floors

    mappings = {
        f"location_{location}": True,
        f"Transaction_{transaction}": True,
        f"Furnishing_{furnishing}": True,
        f"facing_{facing}": True,
        f"Ownership_{ownership}": True,
    }

    unmatched = []
    for col in mappings:
        if col in input_data:
            input_data[col] = 1
        else:
            unmatched.append(col)

    return input_data, unmatched


# -----------------------------
# Single Prediction
# -----------------------------
if st.button("Predict House Price", disabled=floor_error):
    input_data, unmatched = build_input_row(
        carpet_area, bathroom, balcony, current_floor, total_floors,
        location, transaction, furnishing, facing, ownership
    )

    if unmatched:
        st.warning(
            "⚠️ Some selected categories weren't found in the model's feature "
            f"set and were ignored: {', '.join(unmatched)}. The prediction may "
            "be less accurate."
        )

    try:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        prediction = np.expm1(prediction)
        st.success(f"Predicted House Price: ₹ {prediction:,.2f}")

        # Feature importance chart, if the model supports it
        if hasattr(model, "feature_importances_"):
            st.subheader("What influenced this prediction most")
            importances = pd.Series(
                model.feature_importances_, index=feature_names
            ).sort_values(ascending=False).head(10)
            st.bar_chart(importances)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -----------------------------
# Batch Prediction via CSV Upload
# -----------------------------
st.divider()
st.subheader("📄 Batch Prediction (optional)")
st.caption(
    "Upload a CSV with columns: Carpet Area, Bathroom, Balcony, Current_Floor, "
    "Total_Floors, location, Transaction, Furnishing, facing, Ownership"
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        required_cols = [
            "Carpet Area", "Bathroom", "Balcony", "Current_Floor", "Total_Floors",
            "location", "Transaction", "Furnishing", "facing", "Ownership"
        ]
        missing_cols = [c for c in required_cols if c not in batch_df.columns]

        if missing_cols:
            st.error(f"CSV is missing required columns: {', '.join(missing_cols)}")
        else:
            rows = []
            for _, row in batch_df.iterrows():
                input_data, _ = build_input_row(
                    row["Carpet Area"], row["Bathroom"], row["Balcony"],
                    row["Current_Floor"], row["Total_Floors"],
                    row["location"], row["Transaction"], row["Furnishing"],
                    row["facing"], row["Ownership"]
                )
                rows.append(input_data)

            batch_input_df = pd.DataFrame(rows)
            preds = np.expm1(model.predict(batch_input_df))
            batch_df["Predicted Price"] = preds

            st.dataframe(batch_df)
            st.download_button(
                "Download predictions as CSV",
                batch_df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Could not process file: {e}")
