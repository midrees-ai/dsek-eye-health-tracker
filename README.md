# Pre & Post-DSEK Eye Health Tracker

## Project Overview
A Machine Learning web application that predicts IOP Risk and Recovery Stage 
for post-DSEK eye surgery patients. Built on real 3-year medical data of 
Mrs. Zubaida Khan (Grandmother) — 38 hospital visits across 3 hospitals.

## Problem Statement
After DSEK (Descemet Stripping Endothelial Keratoplasty) eye surgery, 
monitoring IOP (Intraocular Pressure) and recovery progress is critical. 
This app predicts risk levels and recovery stages using patient visit data.

## Models Built
- IOP Risk Predictor — Random Forest Classifier — Accuracy: 100%
- Recovery Stage Predictor — Random Forest Classifier — Accuracy: 75%

## Features Used
- Days from surgery
- IOP Left Eye (mmHg)
- Pain level
- Number of medicines
- Medicine types (steroid, antifungal, antiviral, antibiotic, IOP drop)
- Hospital, BCL status, cornea finding, vision, phase

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (Random Forest)
- Streamlit (Web App)
- Matplotlib, Seaborn
- Joblib

## Project Structure
- eye_data.csv — Real patient data (38 visits)
- analysis.py — Exploratory data analysis and charts
- preprocessing.py — Data cleaning and encoding
- model_classification.py — IOP Risk model
- model_recovery.py — Recovery Stage model
- app.py — Streamlit web application

## How to Run
pip install -r requirements.txt
streamlit run app.py

## Developer
Mohammad Idrees
Civil Engineering Graduate — GCET Kashmir
Transitioning into AI/ML
GitHub: midrees-ai