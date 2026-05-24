import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import gdown
import os

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_assets():
    gdown.download(
        "https://drive.google.com/uc?id=13cmNSGFtrc2-X7xGIPIS-4CR4T0cUeue",
        "results.csv", quiet=False
    )
    gdown.download(
        "https://drive.google.com/uc?id=1O6ICDzxiq1B3dOnyxiHHuLBMepkzgRZz",
        "X_test.csv", quiet=False
    )
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    results = pd.read_csv("results.csv")
    X_test  = pd.read_csv("X_test.csv")
    return model, explainer, results, X_test

model, explainer, results, X_test = load_assets()

st.sidebar.title("🔍 Fraud Detection")
page = st.sidebar.radio("Navigate", ["Overview", "Transaction Explorer", "SHAP Explainer"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Filter by Risk Tier**")
selected_tiers = st.sidebar.multiselect(
    "Select tiers",
    options=["Critical Risk", "Suspicious", "Clear"],
    default=["Critical Risk", "Suspicious", "Clear"]
)

filtered = results[results["risk_tier"].isin(selected_tiers)]

# PAGE 1 — OVERVIEW
if page == "Overview":

    st.title("📊 Fraud Operations Overview")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(filtered):,}")
    col2.metric("Total Fraud Cases",  f"{filtered['actual'].sum():,}")
    col3.metric("Detection Rate",     f"{filtered['actual'].mean()*100:.2f}%")
    col4.metric("Avg Fraud Amount",   f"${filtered[filtered['actual']==1]['TransactionAmt'].mean():,.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Tier Distribution")
        tier_counts = filtered["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Tier", "Count"]
        fig = px.pie(tier_counts,
                     names="Risk Tier", values="Count",
                     hole=0.45,
                     color="Risk Tier",
                     color_discrete_map={
                         "Critical Risk": "#E05C5C",
                         "Suspicious":    "#F5A623",
                         "Clear":         "#4A90D9"
                     })
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Fraud Rate by Hour of Day")
        hour_data = filtered.groupby("HourOfDay")["actual"].mean().reset_index()
        hour_data.columns = ["Hour", "Fraud Rate"]
        fig2 = px.line(hour_data, x="Hour", y="Fraud Rate",
                       markers=True,
                       color_discrete_sequence=["#E05C5C"])
        fig2.update_layout(yaxis_tickformat=".2%")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Transaction Amount Distribution by Risk Tier")
    fig3 = px.histogram(filtered, x="TransactionAmt",
                        color="risk_tier", nbins=80,
                        log_y=True, barmode="overlay",
                        color_discrete_map={
                            "Critical Risk": "#E05C5C",
                            "Suspicious":    "#F5A623",
                            "Clear":         "#4A90D9"
                        })
    st.plotly_chart(fig3, use_container_width=True)


# PAGE 2 — TRANSACTION EXPLORER
elif page == "Transaction Explorer":

    st.title("🔎 Transaction Explorer")
    st.markdown("---")

    with st.expander("👉 Don't know a TransactionID? Click here for samples"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("🔴 **Critical Risk**")
            critical_ids = X_test[results["risk_tier"] == "Critical Risk"]["TransactionID"].head(5).values
            for tid in critical_ids:
                st.code(tid)

        with col2:
            st.markdown("🟡 **Suspicious**")
            suspicious_ids = X_test[results["risk_tier"] == "Suspicious"]["TransactionID"].head(5).values
            for tid in suspicious_ids:
                st.code(tid)

        with col3:
            st.markdown("🟢 **Clear**")
            clear_ids = X_test[results["risk_tier"] == "Clear"]["TransactionID"].head(5).values
            for tid in clear_ids:
                st.code(tid)

    search_id = st.text_input("Search by TransactionID", placeholder="e.g. 2987004")

    if search_id:
        match_idx = X_test[X_test["TransactionID"].astype(str) == search_id.strip()].index
        if len(match_idx) > 0:
            row = results.iloc[match_idx[0]]
            st.success("Transaction found!")
            c1, c2, c3 = st.columns(3)
            c1.metric("Risk Score",   f"{row['fraud_probability']:.4f}")
            c2.metric("Risk Tier",    row["risk_tier"])
            c3.metric("Actual Label", "🔴 Fraud" if row["actual"] == 1 else "🟢 Legitimate")
        else:
            st.warning("Transaction ID not found in test set.")

    st.markdown("---")
    st.subheader(f"Showing {len(filtered):,} transactions")

    show_cols = ["TransactionID", "TransactionAmt", "fraud_probability", "risk_tier", "actual"]
    available = [c for c in show_cols if c in filtered.columns]

    st.dataframe(
        filtered[available].sort_values("fraud_probability", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=500
    )


# PAGE 3 — SHAP EXPLAINER
elif page == "SHAP Explainer":

    st.title("🧠 SHAP Transaction Explainer")
    st.markdown("---")

    with st.expander("👉 Don't know a TransactionID? Click here for samples"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("🔴 **Critical Risk**")
            critical_ids = X_test[results["risk_tier"] == "Critical Risk"]["TransactionID"].head(5).values
            for tid in critical_ids:
                st.code(tid)

        with col2:
            st.markdown("🟡 **Suspicious**")
            suspicious_ids = X_test[results["risk_tier"] == "Suspicious"]["TransactionID"].head(5).values
            for tid in suspicious_ids:
                st.code(tid)

        with col3:
            st.markdown("🟢 **Clear**")
            clear_ids = X_test[results["risk_tier"] == "Clear"]["TransactionID"].head(5).values
            for tid in clear_ids:
                st.code(tid)

    txn_input = st.text_input("Enter TransactionID to explain", placeholder="e.g. 2987004")

    if txn_input:
        match = X_test[X_test["TransactionID"].astype(str) == txn_input.strip()]

        if len(match) > 0:
            row_features = match.drop(columns=["TransactionID"]).iloc[[0]]
            prob = model.predict_proba(row_features)[:, 1][0]

            if prob >= 0.75:
                tier, color = "🔴 Critical Risk", "red"
            elif prob >= 0.40:
                tier, color = "🟡 Suspicious", "orange"
            else:
                tier, color = "🟢 Clear", "green"

            st.markdown(f"**Fraud Probability:** `{prob:.4f}`")
            st.markdown(f"**Risk Tier:** :{color}[{tier}]")
            st.markdown("---")

            st.subheader("SHAP Explanation")
            shap_vals = explainer.shap_values(row_features)

            fig, ax = plt.subplots()
            shap.waterfall_plot(
                shap.Explanation(
                    values        = shap_vals[0],
                    base_values   = explainer.expected_value,
                    data          = row_features.iloc[0],
                    feature_names = row_features.columns.tolist()
                ),
                show=False
            )
            st.pyplot(fig)

            st.subheader("Plain English Explanation")
            if prob >= 0.75:
                st.error("This transaction has multiple high-risk signals. "
                         "The model strongly recommends blocking or immediate review.")
            elif prob >= 0.40:
                st.warning("This transaction looks unusual but isn't conclusive. "
                           "Consider adding a soft verification step like OTP.")
            else:
                st.success("This transaction looks normal. "
                           "No significant fraud signals were detected.")
        else:
            st.warning("TransactionID not found.")
