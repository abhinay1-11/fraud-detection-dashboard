# Fraud Detection Dashboard

## Live URL
https://fraud-detection-dashboard-nyqs5t65xuxawnzpiogqnz.streamlit.app

## About
Real-Time Fraud Detection System with Explainable AI and Live Dashboard.
Built as a Week 4 Capstone project for AI & Data Analytics internship.

## Pages
- **Overview** — key metrics, risk tier donut, hour of day fraud rate
- **Transaction Explorer** — search and filter all transactions by TransactionID
- **SHAP Explainer** — enter any TransactionID to get a full AI explanation

## How to run locally
pip install -r requirements.txt
streamlit run app.py

## Tech Stack
- LightGBM — fraud detection model
- SHAP — explainable AI
- Streamlit — dashboard
- Plotly — interactive charts
- Google Drive — large file storage
