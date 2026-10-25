# Fraud Detection with Explainable AI

An interactive dashboard that flags suspicious transactions and explains each prediction with SHAP.

**[Live demo](https://fraud-detection-dashboard-nyqs5t65xuxawnzpiogqnz.streamlit.app)** · Python · LightGBM · SHAP · Streamlit

Built during my Data Science internship at **XylofiAI** (Apr – May 2026).

## What it does
- Analyzes 590K+ transactions to detect fraud on heavily imbalanced data
- Trains a LightGBM classifier to flag suspicious transactions
- Explains predictions with SHAP so every flagged transaction has a reason
- Presents everything in a 3-page Streamlit dashboard

## Dashboard pages
- **Overview:** key metrics, risk tier breakdown and fraud rate by hour of day
- **Transaction Explorer:** search and filter transactions by TransactionID
- **SHAP Explainer:** enter any TransactionID to see a full explanation of the prediction

## Tech stack
LightGBM · SHAP · Streamlit · Plotly · Pandas

## Run locally
pip install -r requirements.txt
streamlit run app.py
