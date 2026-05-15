import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="TSLA Predictor", layout="wide")

st.title("TSLA Stock Direction Predictor")
st.caption("XGBoost + FinBERT Sentiment + Macro Features")

# ── Metrics row ───────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Accuracy", "52.1%", "+2.1% vs random")
col2.metric("Average AUC",      "0.533",  "+0.033 vs baseline")
col3.metric("Average F1",       "0.509")
col4.metric("Total Features",   "35")

st.divider()

# ── Fold results chart ────────────────────────────────────────────
st.subheader("Walk-forward fold results")

fold_data = pd.DataFrame({
    'Fold':     [1, 2, 3, 4, 5],
    'Accuracy': [0.5720, 0.4680, 0.5160, 0.5280, 0.5200],
    'AUC':      [0.5654, 0.5075, 0.5507, 0.5432, 0.5000],
    'F1':       [0.5342, 0.4069, 0.5189, 0.5202, 0.5146],
})

fig = go.Figure()
fig.add_trace(go.Scatter(x=fold_data['Fold'], y=fold_data['Accuracy'],
              name='Accuracy', mode='lines+markers', line=dict(color='#378ADD')))
fig.add_trace(go.Scatter(x=fold_data['Fold'], y=fold_data['AUC'],
              name='AUC', mode='lines+markers', line=dict(color='#1D9E75')))
fig.add_trace(go.Scatter(x=fold_data['Fold'], y=fold_data['F1'],
              name='F1', mode='lines+markers', line=dict(color='#EF9F27')))
fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
              annotation_text="Random baseline")
fig.update_layout(yaxis_range=[0.38, 0.62], xaxis_title="Fold",
                  yaxis_title="Score", height=350)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Feature importance ────────────────────────────────────────────
st.subheader("Top 10 feature importances")

importance_df = pd.DataFrame({
    'feature':    ['RSI','MACD','sentiment_lag1','VIX_lag1','BB_pct_b',
                   'ROC_5','Close_vs_SMA20','Volatility_20','SPY_ret_lag1','Volume_ratio'],
    'importance': [0.12, 0.10, 0.09, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05]
})

fig2 = px.bar(importance_df.sort_values('importance'),
              x='importance', y='feature', orientation='h',
              color_discrete_sequence=['#378ADD'])
fig2.update_layout(height=350, xaxis_title="Importance", yaxis_title="")
st.plotly_chart(fig2, use_container_width=True)
