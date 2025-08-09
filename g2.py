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
    features = joblib.load("model_features_xgb.pkl")     # ordered list
    y_info = joblib.load("y_transform_xgb.pkl")          # {"transform":"log1p","inverse":"expm1"}
    return model, features, y_info

model, MODEL_FEATURES, Y_INFO = load_artifacts()
MODEL_FEATURES = list(MODEL_FEATURES)  # ensure list

# -----------------------------
# 2) Helpers
# -----------------------------
def align_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only model features, add missing with 0, order columns."""
    df = df.copy()
    # keep numeric only (model trained on numeric)
    df = df.select_dtypes(include=[np.number])

    # drop extras, add missings
    for col in set(df.columns) - set(MODEL_FEATURES):
        df.drop(columns=[col], inplace=True)
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    # ensure order and numeric dtype
    df = df[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    return df

def inverse_transform(y_hat_log: np.ndarray) -> np.ndarray:
    inv = (Y_INFO or {}).get("inverse", "expm1")
    if inv == "expm1":
        return np.expm1(y_hat_log)
    # fallback (identity)
    return y_hat_log

def predict_df(df_in: pd.DataFrame) -> pd.Series:
    X = align_features(df_in)
    yhat_log = model.predict(X)
    yhat = inverse_transform(yhat_log)
    return pd.Series(yhat, index=X.index, name="Predicted_Total_Affected")

# -----------------------------
# 3) UI
# -----------------------------
st.title("🌍 Disaster Impact Predictor (XGBoost)")

mode = st.radio("Choose input mode:", ["Upload CSV (batch)", "Enter a single record"], horizontal=True)

with st.expander("Model details", expanded=False):
    st.write(f"- Features expected: *{len(MODEL_FEATURES)}*")
    st.code(", ".join(MODEL_FEATURES[:30]) + ("..." if len(MODEL_FEATURES) > 30 else ""))
    st.write("Target transform:", Y_INFO)

if mode == "Upload CSV (batch)":
    st.subheader("Batch Prediction")
    up = st.file_uploader("Upload a CSV containing the predictor columns", type=["csv"])
    if up:
        df = pd.read_csv(up)
        st.write("Preview:", df.head())
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
        pred = predict_df(df_one).iloc[0]
        st.metric("Predicted Total Affected", f"{pred:,.0f}")