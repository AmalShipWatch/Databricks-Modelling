import streamlit as st
import requests
import base64
from pathlib import Path

TOKEN = st.secrets["DATABRICKS_TOKEN"]
URL = "https://dbc-f2ea18fc-2f89.cloud.databricks.com/serving-endpoints/vessel_consumption/invocations"

st.set_page_config(page_title="Vessel Fuel Predictor", page_icon="🚢", layout="centered")

# Load and encode background image as base64
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64_image(".streamlit/static/shipwatch.jpg")

st.markdown(f"""
    <style>
        /* ---- hide streamlit ui ---- */
        #MainMenu {{visibility: hidden;}}
        header {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stAppDeployButton {{display: none;}}
        [data-testid="stToolbar"] {{visibility: hidden;}}
        [data-testid="stDecoration"] {{display: none;}}
        [data-testid="stStatusWidget"] {{display: none;}}

        /* ---- background image with blur ---- */
        .stApp {{
            background: url("data:image/jpeg;base64,{img_base64}") no-repeat center center fixed;
            background-size: cover;
        }}
        .stApp::before {{
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: inherit;
            filter: blur(12px) brightness(1.0);
            z-index: 0;
        }}

        /* ---- main container card ---- */
        .block-container {{
            position: relative;
            z-index: 1;
            background: rgba(0, 0, 0, 0.92) !important;
            border-radius: 40px;
            padding: 2.5rem 2rem 2rem 2rem !important;
            max-width: 650px !important;
            margin-top: 60px !important;
            box-shadow: 0 4px 32px rgba(255,255,255,0.4);
        }}

        /* ---- title ---- */
        h1 {{
            color: #80c41c !important;
            text-align: center;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 1.6rem;
            margin-bottom: 1.5rem;
        }}

        /* ---- labels ---- */
        label, .stNumberInput label {{
            color: #80c41c !important;
            font-weight: 600 !important;
            font-family: 'Segoe UI', Arial, sans-serif;
        }}

        /* ---- input fields ---- */
        input[type="number"] {{
            background: #101c36 !important;
            border: 2px solid #80c41c !important;
            color: #fcfcfc !important;
            border-radius: 8px !important;
            font-size: 1rem !important;
        }}
        input[type="number"]:focus {{
            border-color: #80c41c !important;
            box-shadow: 0 0 0 2px rgba(128, 196, 28, 0.35) !important;
            background: #18244a !important;
        }}

        /* ---- predict button ---- */
        .stButton > button {{
            background: #80c41c !important;
            color: #071447 !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            padding: 10px 28px !important;
            width: 100%;
            margin-top: 8px;
            transition: background 0.2s, color 0.2s;
        }}
        .stButton > button:hover {{
            background: #071447 !important;
            color: #fcfcfc !important;
            border: 2px solid #80c41c !important;
        }}

        /* ---- success / error boxes ---- */
        .stSuccess {{
            background: #eaf6d5 !important;
            color: #071447 !important;
            border-left: 4px solid #80c41c !important;
            border-radius: 8px !important;
        }}
        .stAlert {{
            border-radius: 8px !important;
        }}
    </style>
""", unsafe_allow_html=True)

st.title("🚢 Consumption Predictor")
st.markdown("<p style='color:#cdd6f4; text-align:center; margin-top:-12px; font-size:1.25rem;'>Enter vessel parameters to get a predicted fuel consumption</p>", unsafe_allow_html=True)

power = st.number_input("Power (kW)", min_value=0.0, value=14.17, format="%.4f")
draft = st.number_input("Draft (m)", min_value=0.0, max_value=30.0, value=7.0, step=0.5, format="%.1f")

if st.button("Predict"):
    payload = {"inputs": [[power, draft]]}
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