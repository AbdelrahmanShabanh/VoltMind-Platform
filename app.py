import datetime
import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Smart Grid Load Forecaster",
    page_icon="⚡",
    layout="centered",
)

# Main title
st.title("⚡ Smart Grid Load Forecasting")
st.markdown("### Elite Hexa Team - AI Prediction Dashboard")
st.write(
    "Enter the parameters below to predict the required active power for the next 5 minutes."
)

# Load the trained model
@st.cache_resource
def load_model_correctly():
    return joblib.load("xgboost_smartgrid_model.pkl")


model = load_model_correctly()

# Split the page into two columns for better input organization
col1, col2 = st.columns(2)

with col1:
    st.subheader("Time Features")
    input_time = st.time_input(
        "Select Time",
        datetime.time(20, 0),  # Default time: 8:00 PM
    )
    is_weekend = st.selectbox(
        "Is it Weekend?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

with col2:
    st.subheader("Grid Features")
    avg_voltage = st.number_input(
        "Average Voltage (V)",
        min_value=200.0,
        max_value=250.0,
        value=230.0,
    )
    lag_1 = st.number_input("Power 5 mins ago (kW)", value=3.5)
    lag_2 = st.number_input("Power 10 mins ago (kW)", value=3.4)
    rolling_mean = st.number_input("Rolling Mean (Last 15 mins)", value=3.45)

# Prediction button
if st.button("🔮 Predict Grid Load"):
    # Prepare the input data in the same order used during model training
    input_data = pd.DataFrame(
        {
            "rolling_mean_3": [rolling_mean],
            "lag_1_power": [lag_1],
            "lag_2_power": [lag_2],
            "avg_voltage": [avg_voltage],
            "minute": [input_time.minute],
            "hour": [input_time.hour],
            "dayofweek": [day_of_week],
            "is_weekend": [is_weekend],
        }
    )

    # Generate the prediction
    prediction = model.predict(input_data)[0]

    # Display the prediction result
    st.success(f"### ⚡ Predicted Active Power: {prediction:.3f} kW")
    st.info(
        "The grid should allocate this amount of power to avoid blackouts or overloads."
    )