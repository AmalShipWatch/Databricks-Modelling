import streamlit as st
import requests

TOKEN = st.secrets["DATABRICKS_TOKEN"]
URL = "https://dbc-f2ea18fc-2f89.cloud.databricks.com/serving-endpoints/vessel_consumption/invocations"

st.set_page_config(page_title="Vessel Fuel Predictor", page_icon="🚢")
st.title("🚢 Vessel Fuel Consumption Predictor")

st.markdown("Enter vessel parameters below to get a predicted fuel consumption.")

power = st.number_input("Power (kW)", min_value=0.0, max_value=17, value=14.17, step=0.01, format="%.2f")
draft = st.number_input("Draft (m)", min_value=0.0, max_value=30.0, value=7.0, step=0.5, format="%.1f")

if st.button("Predict"):
    payload = {"inputs": [[power, draft]]}  # 👈 now passing both
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    with st.spinner("Calling model..."):
        response = requests.post(URL, headers=headers, json=payload)

    try:
        result = response.json()
        prediction = result["predictions"][0][0]
        st.success(f"Predicted Fuel Consumption: **{prediction:.4f}**")
    except Exception as e:
        st.error(f"Something went wrong: {response.text}")