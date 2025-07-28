import streamlit as st
import torch
import numpy as np
import joblib
from model import InsuranceModel

# Load the scaler
scaler = joblib.load("scaler.pkl")

# Load the model
INPUT_SIZE = 8  # age, sex, bmi, children, smoker, region_northwest, southeast, southwest
model = InsuranceModel(INPUT_SIZE)
model.load_state_dict(torch.load("insurance_model.pth", map_location=torch.device("cpu")))
model.eval()

# Streamlit page setup
st.set_page_config(page_title="Dynamic Insurance Pricing", page_icon="💸")
st.title("💸 Dynamic Insurance Pricing Predictor")
st.markdown("Estimate medical insurance charges based on customer details.")

# User input section
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0, step = 0.1, format="%.1f")
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Encode categorical values
sex_encoded = 1 if sex == "male" else 0
smoker_encoded = 1 if smoker == "yes" else 0

# One-hot encode region (drop_first=True used during training, so drop 'northeast')
region_encoded = [0, 0, 0]  # Order: northwest, southeast, southwest
region_map = {"northwest": 0, "southeast": 1, "southwest": 2}
if region != "northeast":
    region_encoded[region_map[region]] = 1

# Prepare input
input_data = [age, sex_encoded, bmi, children, smoker_encoded] + region_encoded
input_array = np.array(input_data).reshape(1, -1)

# Apply scaling
input_scaled = scaler.transform(input_array)

# Predict
if st.button("Predict Insurance Charges"):
    with torch.no_grad():
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        log_prediction = model(input_tensor).item()
        predicted_charges = np.exp(log_prediction)  # Reverse log transformation

    st.success(f"✅ Estimated Insurance Charges: **${predicted_charges:,.2f}**")
    st.caption("This estimate is based on a model trained on U.S. health insurance data.")
