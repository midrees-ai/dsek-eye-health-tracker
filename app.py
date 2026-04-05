import streamlit as st
import numpy as np
import joblib

# Load models
iop_model = joblib.load('iop_risk_model.pkl')
rec_model = joblib.load('recovery_model.pkl')

st.title("Pre & Post-DSEK Eye Health Tracker")
st.markdown("### Corneal Edema Recovery & IOP Risk Predictor")
st.markdown("Patient: Mrs. Zubaida Khan | Real Medical Data | 38 Visits")

st.sidebar.header("Enter Patient Data")

days = st.sidebar.slider("Days From Surgery", 0, 1000, 100)
iop_left = st.sidebar.slider("IOP Left Eye (mmHg)", 5, 40, 15)
pain_level = st.sidebar.slider("Pain Level (0-10)", 0, 10, 2)
num_medicines = st.sidebar.slider("Number of Medicines", 0, 10, 3)
has_antifungal = st.sidebar.selectbox("Has Antifungal", [0, 1])
has_antiviral = st.sidebar.selectbox("Has Antiviral", [0, 1])
has_iop_drop = st.sidebar.selectbox("Has IOP Drop", [0, 1])
has_steroid = st.sidebar.selectbox("Has Steroid", [0, 1])
has_antibiotic = st.sidebar.selectbox("Has Antibiotic", [0, 1])
hospital_encoded = st.sidebar.selectbox("Hospital (0/1/2)", [0, 1, 2])
vision_encoded = st.sidebar.selectbox("Vision Level (0/1/2)", [0, 1, 2])
cornea_encoded = st.sidebar.selectbox("Cornea Finding (0/1/2)", [0, 1, 2])
bcl_status_encoded = st.sidebar.selectbox("BCL Status (0/1)", [0, 1])
phase_encoded = st.sidebar.selectbox("Phase (0/1/2/3)", [0, 1, 2, 3])

input_data = np.array([[days, iop_left, pain_level, num_medicines,
                        has_antifungal, has_antiviral, has_iop_drop,
                        has_steroid, has_antibiotic, hospital_encoded,
                        vision_encoded, cornea_encoded,
                        bcl_status_encoded, phase_encoded]])

if st.button("Predict"):
    iop_pred = iop_model.predict(input_data)[0]
    rec_pred = rec_model.predict(input_data)[0]

    st.subheader("Results")

    if iop_pred == 1:
        st.error("IOP Risk: HIGH RISK ⚠️")
    else:
        st.success("IOP Risk: Normal ✅")

    stage_map = {0: "Early Recovery", 1: "Mid Recovery",
                 2: "Good Recovery", 3: "Pre Surgery"}
    st.info(f"Recovery Stage: {stage_map.get(rec_pred, rec_pred)}")