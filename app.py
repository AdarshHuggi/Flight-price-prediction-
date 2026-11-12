"""Simple Streamlit interface for the trained flight fare model."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from train_model import DATA_PATH, MODEL_PATH, load_data

st.set_page_config(page_title="Flight Fare Predictor", page_icon="✈️", layout="centered")
st.title("✈️ Flight fare predictor")
st.write("Estimate a fare from the project's historical 2021 flight data.")

if not MODEL_PATH.exists():
    st.error("The model has not been trained yet. Run `python train_model.py` first.")
    st.stop()

@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def get_data():
    return load_data()

bundle = get_model()
data = get_data()

with st.form("prediction"):
    left, right = st.columns(2)
    airline = left.selectbox("Airline", sorted(data.Airline.dropna().unique()))
    source = right.selectbox("From", sorted(data.Source.dropna().unique()))
    destination = left.selectbox("To", sorted(data.Destination.dropna().unique()))
    month = right.selectbox("Travel month", list(data.Month.dropna().unique()))
    day = left.selectbox("Day of month", range(1, 32), index=20)
    route_options = sorted(data.loc[(data.Source == source) & (data.Destination == destination), "Route"].dropna().unique())
    route = right.selectbox("Route", route_options or sorted(data.Route.dropna().unique()))
    dep_hour = left.slider("Departure hour", 0, 23, 9)
    dep_min = right.selectbox("Departure minute", [0, 15, 30, 45])
    duration_hour = left.number_input("Duration (hours)", min_value=0, max_value=30, value=2)
    duration_min = right.selectbox("Additional duration minutes", [0, 15, 30, 45])
    stops = st.selectbox("Stops", sorted(data.stop.dropna().unique()))
    submitted = st.form_submit_button("Estimate fare", type="primary", use_container_width=True)

if submitted:
    # Derive a plausible arrival time from departure plus duration.
    arrival_total = dep_hour * 60 + dep_min + int(duration_hour) * 60 + duration_min
    row = pd.DataFrame([{
        "Airline": airline, "Source": source, "Destination": destination, "Month": month,
        "Route": route, "stop": stops, "Day": day, "Dep_hour": dep_hour, "Dep_min": dep_min,
        "Dur_hour": duration_hour, "Dur_min": duration_min,
        "Arr_hour": (arrival_total // 60) % 24, "Arr_min": arrival_total % 60,
    }])
    estimate = float(bundle["model"].predict(row[bundle["features"]])[0])
    st.metric("Estimated fare", f"₹{estimate:,.0f}")
    st.caption("Estimate based on historical 2021 data. Actual fares change and may differ substantially.")

with st.expander("Model evaluation"):
    metrics = bundle["metrics"]
    st.write(f"Trained on {metrics['rows']:,} rows; held-out MAE ₹{metrics['mae']:,.0f}, RMSE ₹{metrics['rmse']:,.0f}, R² {metrics['r2']:.3f}.")
    st.caption(metrics["note"])
