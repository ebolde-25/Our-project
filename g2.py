import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Disaster Impact – XGBoost Predictor", layout="wide")

# -----------------------------
# 1) Load artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("xgb_model.pkl")                 # Pipeline(imputer + xgb)
    features = joblib.load("model_features_xgb.pkl")     # ordered list of features
    y_info = joblib.load("y_transform_xgb.pkl")          # {"transform":"log1p","inverse":"expm1"}
    return model, list(features), y_info

model, MODEL_FEATURES, Y_INFO = load_artifacts()

# -----------------------------
# 2) Helpers
# -----------------------------
def sanitize_to_model_X(df_in: pd.DataFrame) -> pd.DataFrame:
    """Align uploaded dataframe to model's expected features and convert to numeric."""
    df = df_in.copy()
    df.columns = df.columns.str.strip()  # remove whitespace in headers

    # Add missing expected columns with 0
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # Keep only expected columns, in correct order
    X = df[MODEL_FEATURES].copy()

    # Convert to numeric, forcing invalid values to NaN (imputer will handle)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Diagnostics for the user
    extra_cols = [c for c in df_in.columns if c not in MODEL_FEATURES]
    if extra_cols:
        st.info(f"Ignoring unexpected columns: {extra_cols[:15]}{' ...' if len(extra_cols) > 15 else ''}")
    if X.isna().any().any():
        st.warning("Some non-numeric or blank values were replaced with NaN and will be imputed by the model.")

    return X

def inverse_transform(yhat_log: np.ndarray) -> np.ndarray:
    if (Y_INFO or {}).get("inverse") == "expm1":
        return np.expm1(yhat_log)
    return yhat_log

def predict_df(df_in: pd.DataFrame) -> pd.Series:
    X = sanitize_to_model_X(df_in)
    yhat_log = model.predict(X)
    yhat = inverse_transform(yhat_log)
    return pd.Series(yhat, index=X.index, name="Predicted_Total_Affected")

# -----------------------------
# 3) UI
# -----------------------------
st.title("🌍 Disaster Impact Predictor (XGBoost)")

mode = st.radio("Choose input mode:", ["Upload CSV (batch)", "Enter a single record"], horizontal=True)

with st.expander("Model details", expanded=False):
    st.write(f"- Features expected: {len(MODEL_FEATURES)}")
    st.code(", ".join(MODEL_FEATURES[:30]) + ("..." if len(MODEL_FEATURES) > 30 else ""))
    st.write("Target transform:", Y_INFO)
    # Template download button
    template_df = pd.DataFrame(columns=MODEL_FEATURES)
    st.download_button(
        label="Download CSV Template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="disaster_prediction_template.csv",
        mime="text/csv"
    )

# -----------------------------
# 4) Modes
# -----------------------------
if mode == "Upload CSV (batch)":
    st.subheader("Batch Prediction")
    up = st.file_uploader("Upload a CSV containing the predictor columns", type=["csv"])
    if up:
        df = pd.read_csv(up)
        st.write("Preview:", df.head())
        try:
            preds = predict_df(df)
            out = df.copy()
            out["Predicted_Total_Affected"] = preds
            st.success("Done! Download your results below.")
            st.download_button(
                "Download predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error("Could not generate predictions. Please check your file format.")
            st.write(str(e))
    else:
        st.info("Upload a CSV with the exact columns the model expects, or download the template above.")

else:
    st.subheader("Single Prediction")
    st.caption("Enter values for the model’s features (numbers only). Leave blank = 0.")
    cols = st.columns(3)
    inputs = {}
    for i, feat in enumerate(MODEL_FEATURES):
        with cols[i % 3]:
            val = st.text_input(feat, value="")
            try:
                inputs[feat] = float(val) if val.strip() != "" else 0.0
            except:
                inputs[feat] = 0.0

    if st.button("Predict"):
        df_one = pd.DataFrame([inputs])
        try:
            pred = predict_df(df_one).iloc[0]
            st.metric("Predicted Total Affected", f"{pred:,.0f}")
        except Exception as e:
            st.error("Prediction failed.")
            st.write(str(e))
