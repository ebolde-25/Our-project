import streamlit as st 
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------
st.set_page_config(page_title="Disaster Impact Predictor - Group 2", layout="wide")

# ------------------------------------------------
# LOAD TRAINED ARTIFACTS (XGBoost + features)
# ------------------------------------------------
# Expect these files in the same folder as app.py
MODEL_PATH = "xgb_disaster_model.pkl"      # trained on log1p(target)
FEATS_PATH = "model_features.pkl"          # list of feature names (strings)

model = joblib.load(MODEL_PATH)
X_columns = joblib.load(FEATS_PATH)

# ------------------------------------------------
# SIDEBAR NAVIGATION (same layout vibe as your sample)
# ------------------------------------------------
with st.sidebar:
    st.markdown(
        "<h3 style='margin-bottom: 10px; color: #92400e;'>📑 Navigation</h3>",
        unsafe_allow_html=True
    )
    selected = option_menu(
        menu_title=None,
        options=["Home", "Predictor", "About"],
        icons=["house", "bar-chart", "info-circle"],
        default_index=1,
        styles={
            "container": {"padding": "0!important", "background-color": "#fef3c7"},
            "icon": {"color": "#92400e", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "color": "#000000",
                "--hover-color": "#fde68a"
            },
            "nav-link-selected": {
                "background-color": "#fcd34d",
                "color": "#000000"
            },
        }
    )

# ------------------------------------------------
# HOME TAB
# ------------------------------------------------
if selected == "Home":
    st.title("🏠 Welcome to the Disaster Impact Predictor - Group 2")
    st.write(
        "This tool estimates the number of people potentially affected by a disaster "
        "using a tuned XGBoost model trained on historical records."
    )

# ------------------------------------------------
# PREDICTOR TAB
# ------------------------------------------------
elif selected == "Predictor":
    st.title("📊 Disaster Impact Prediction")
    st.write("Provide the details to estimate the *Total Affected*.")

    # If your training included categorical encodings, place your mappings here.
    # Leave empty if your model used numeric-only features.
    disaster_group_options = {"Biological": 0, "Climatological": 1, "Geophysical": 2, "Hydrological": 3, "Meteorological": 4, "Technological": 5}
    disaster_type_options  = {"Flood": 0, "Earthquake": 1, "Storm": 2, "Epidemic": 3}
    country_options        = {"Ghana": 0, "Nigeria": 1, "Kenya": 2, "South Africa": 3}
    region_options         = {"Africa": 0, "Asia": 1, "Europe": 2, "Americas": 3}

    # Build inputs dynamically from the saved feature list
    vals = {}
    for col in X_columns:
        label = col.replace("_", " ").title()
        low = col.lower()

        if low == "year":
            vals[col] = st.number_input(label, min_value=1900, max_value=2100, value=2023, step=1)

        elif low == "disaster_group":
            choice = st.selectbox(label, list(disaster_group_options.keys()))
            vals[col] = disaster_group_options[choice]

        elif low == "disaster_type":
            choice = st.selectbox(label, list(disaster_type_options.keys()))
            vals[col] = disaster_type_options[choice]

        elif low == "country":
            choice = st.selectbox(label, list(country_options.keys()))
            vals[col] = country_options[choice]

        elif low == "region":
            choice = st.selectbox(label, list(region_options.keys()))
            vals[col] = region_options[choice]

        # Common numeric features from your PDF (edit as needed)
        elif low in ["total_deaths", "number_injured", "number_affected", "number_homeless"]:
            vals[col] = st.number_input(label, min_value=0.0, value=0.0, step=1.0)

        else:
            # Default numeric input for any other feature
            vals[col] = st.number_input(label, value=0.0, step=1.0)

    # Convert to DataFrame in the exact column order expected by the model
    input_df = pd.DataFrame([vals], columns=X_columns).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Predict
    if st.button("Predict Affected People"):
        try:
            # Model was trained on log1p(target), so we inverse-transform with expm1
            log_pred = model.predict(input_df.values)
            pred = np.expm1(log_pred)  # back to original scale
            st.success(f"📌 Estimated Number of People Affected: {float(pred[0]):,.0f}")
        except Exception as e:
            st.error("🚫 Prediction failed.")
            st.code(str(e))
            st.write("Expected columns (training order):", X_columns)
            st.write("Current input shape:", input_df.shape)

# ------------------------------------------------
# ABOUT TAB
# ------------------------------------------------
elif selected == "About":
    st.title("ℹ About This App")
    st.markdown("""
*Model*: Tuned XGBoost Regressor trained on log1p(Total Affected) and numeric features.  
*Artifacts*: xgb_disaster_model.pkl, model_features.pkl (saved with joblib).  
*Prediction*: We apply expm1 on the model output to return to the original scale.  
*Purpose*: Support preparedness and resource planning with quick, consistent estimates.
""")
