# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import os
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ── STREAMLIT PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(page_title="TSLA Predictor", layout="wide")

st.title("TSLA Stock Direction Predictor")
st.caption("XGBoost + FinBERT Sentiment + Macro Features Pipeline")

# ── DATA PROCESSING BACKEND ─────────────────────────────────────────────────

@st.cache_data(show_spinner="Downloading stock data and processing features...")
def load_and_process_data():
    # Download full TSLA history
    raw = yf.download('TSLA', start='2019-01-01', end='2024-12-31', progress=False, auto_adjust=True)
    raw = raw.reset_index()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] if col[1] == '' else col[0] for col in raw.columns]

    raw['Date'] = pd.to_datetime(raw['Date']).dt.tz_localize(None)
    raw = raw.sort_values('Date').reset_index(drop=True)
    raw['Adj Close'] = raw['Close']

    # Load Sentiment Data (Looking for the file locally in the repository)
    csv_path = 'TSLA_scored.csv'
    if not os.path.exists(csv_path):
        st.error(f"Critical Error: '{csv_path}' not found in the directory. Please upload it to GitHub alongside this app script.")
        st.stop()
        
    news_raw = pd.read_csv(csv_path)
    news_raw['Date'] = pd.to_datetime(news_raw['Date'])

    daily = news_raw.groupby('Date').agg(
        sent_pos_mean  = ('positive','mean'),
        sent_neg_mean  = ('negative','mean'),
        sent_pos_max   = ('positive','max'),
        sent_neg_max   = ('negative','max'),
        headline_count = ('headline','count')
    ).reset_index()

    daily['sent_net']      = daily['sent_pos_mean'] - daily['sent_neg_mean']
    daily = daily.sort_values('Date').reset_index(drop=True)
    daily['sent_net_roll3'] = daily['sent_net'].rolling(3).mean()
    daily['sent_pos_roll3'] = daily['sent_pos_mean'].rolling(3).mean()

    SENT_COLS = ['sent_pos_mean','sent_neg_mean','sent_net',
                 'sent_pos_max','sent_neg_max','headline_count',
                 'sent_net_roll3','sent_pos_roll3']
    sent_lag = daily[['Date'] + SENT_COLS].copy()
    for c in SENT_COLS:
        sent_lag[c] = sent_lag[c].shift(1)
    sent_lag.columns = ['Date'] + [f'{c}_lag1' for c in SENT_COLS]

    df = raw.copy()

    # Technical features
    for lag in [1, 2, 3]:
        df[f'Close_lag{lag}'] = df['Close'].shift(lag)
    df['Open_lag1']   = df['Open'].shift(1)
    df['High_lag1']   = df['High'].shift(1)
    df['Low_lag1']    = df['Low'].shift(1)
    df['Volume_lag1'] = df['Volume'].shift(1)

    df['SMA_5']  = df['Close_lag1'].rolling(5).mean()
    df['SMA_20'] = df['Close_lag1'].rolling(20).mean()
    df['EMA_10'] = df['Close_lag1'].ewm(span=10, adjust=False).mean()

    df['Close_vs_SMA5']  = df['Close_lag1'] / df['SMA_5']  - 1
    df['Close_vs_SMA20'] = df['Close_lag1'] / df['SMA_20'] - 1
    df['SMA5_vs_SMA20']  = df['SMA_5'] / df['SMA_20'] - 1

    delta = df['Close_lag1'].diff()
    gain  = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    bb_mid = df['Close_lag1'].rolling(20).mean()
    bb_std = df['Close_lag1'].rolling(20).std()
    df['BB_pct_b'] = (df['Close_lag1'] - (bb_mid - 2*bb_std)) / (4*bb_std)
    df['BB_width'] = (4*bb_std) / bb_mid

    ema12 = df['Close_lag1'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close_lag1'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']

    df['ROC_1']  = df['Close_lag1'].pct_change(1)
    df['ROC_5']  = df['Close_lag1'].pct_change(5)
    df['ROC_10'] = df['Close_lag1'].pct_change(10)

    log_ret = np.log(df['Close_lag1'] / df['Close_lag1'].shift(1))
    df['Volatility_5']  = log_ret.rolling(5).std()
    df['Volatility_20'] = log_ret.rolling(20).std()

    df['Volume_SMA10'] = df['Volume_lag1'].rolling(10).mean()
    df['Volume_ratio'] = df['Volume_lag1'] / df['Volume_SMA10']
    df['HL_range']     = (df['High_lag1'] - df['Low_lag1']) / df['Close_lag1']
    df['Body']         = (df['Close_lag1'] - df['Open_lag1']) / df['Close_lag1']

    # Macro features
    vix = yf.download('^VIX', start='2010-01-01', end='2024-12-31', progress=False, auto_adjust=True)[['Close']].reset_index()
    spy = yf.download('SPY',  start='2010-01-01', end='2024-12-31', progress=False, auto_adjust=True)[['Close']].reset_index()
    vix.columns = ['Date','VIX']
    spy.columns = ['Date','SPY']
    for d in [vix, spy]:
        d['Date'] = pd.to_datetime(d['Date']).dt.tz_localize(None)
    vix['VIX_lag1']        = vix['VIX'].shift(1)
    vix['VIX_change_lag1'] = vix['VIX'].pct_change().shift(1)
    spy['SPY_ret_lag1']    = spy['SPY'].pct_change().shift(1)
    spy['SPY_ret5_lag1']   = spy['SPY'].pct_change(5).shift(1)

    df = df.merge(vix[['Date','VIX_lag1','VIX_change_lag1']], on='Date', how='left')
    df = df.merge(spy[['Date','SPY_ret_lag1','SPY_ret5_lag1']], on='Date', how='left')

    # Sentiment merging
    sent_lag['Date'] = pd.to_datetime(sent_lag['Date'])
    df = df.merge(sent_lag, on='Date', how='left')
    sent_feature_cols = [c for c in df.columns if 'sent_' in c or 'headline_count' in c]
    df[sent_feature_cols] = df[sent_feature_cols].fillna(0)

    # Target assignment
    df['Target'] = ((df['Adj Close'].shift(-1) / df['Adj Close'] - 1) > 0.005).astype(int)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    
    return df, SENT_COLS

# Load execution data
df, SENT_COLS = load_and_process_data()

# Feature Lists setup
TECH_FEATURES  = [
    'Close_lag1','Close_lag2','Close_lag3',
    'Open_lag1','High_lag1','Low_lag1','Volume_lag1',
    'SMA_5','SMA_20','EMA_10',
    'Close_vs_SMA5','Close_vs_SMA20','SMA5_vs_SMA20',
    'RSI','BB_pct_b','BB_width',
    'MACD','MACD_signal','MACD_hist',
    'ROC_1','ROC_5','ROC_10',
    'Volatility_5','Volatility_20',
    'Volume_ratio','HL_range','Body'
]
MACRO_FEATURES = ['VIX_lag1','VIX_change_lag1','SPY_ret_lag1','SPY_ret5_lag1']
SENT_FEATURES  = [f'{c}_lag1' for c in SENT_COLS if f'{c}_lag1' in df.columns]
ALL_FEATURES   = TECH_FEATURES + MACRO_FEATURES + SENT_FEATURES

X = df[ALL_FEATURES].copy()
y = df['Target'].copy()
spw = (y==0).sum() / (y==1).sum()

# ── MODEL TRAINING ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Running Backtest (Walk-Forward Validation)...")
def train_and_backtest(X, y, spw):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            scale_pos_weight=spw,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric='logloss', random_state=42, n_jobs=-1
        ))
    ])

    tscv = TimeSeriesSplit(n_splits=5, test_size=250)
    fold_accuracies, fold_aucs, fold_f1s = [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < 500:
            continue

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        fold_accuracies.append(accuracy_score(y_test, preds))
        fold_f1s.append(f1_score(y_test, preds, average='weighted'))
        fold_aucs.append(roc_auc_score(y_test, probs))

    # Fit final model on last seen historical fold chunk for feature importance
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    importances = model.named_steps['clf'].feature_importances_

    return fold_accuracies, fold_aucs, fold_f1s, importances

accs, aucs, f1s, importances = train_and_backtest(X, y, spw)

# ── FRONTEND METRICS ROW ────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Accuracy", f"{np.mean(accs)*100:.1f}%", f"{(np.mean(accs)-0.5)*100:+.1f}% vs random")
col2.metric("Average AUC", f"{np.mean(aucs):.3f}", f"{np.mean(aucs)-0.5:+.3f} vs baseline")
col3.metric("Average F1 Score", f"{np.mean(f1s):.3f}")
col4.metric("Total Features Trained", f"{len(ALL_FEATURES)}")

st.divider()

# ── WALK-FORWARD GRAPH ──────────────────────────────────────────────────────
st.subheader("Walk-forward Validation Performance (By Fold)")

fold_data = pd.DataFrame({
    'Fold': [f"Fold {i+1}" for i in range(len(accs))],
    'Accuracy': accs,
    'AUC': aucs,
    'F1': f1s,
})

fig = go.Figure()
fig.add_trace(go.Scatter(x=fold_data['Fold'], y=fold_data['Accuracy'], name='Accuracy', mode='lines+markers', line=dict(color='#378ADD')))
fig.add_trace(go.Scatter(x=fold_data['Fold'], y=fold_data['AUC'], name='AUC', mode='lines+markers', line=dict(color='#1D9E75')))
fig.add_trace(go.Scatter(x=fold_data['Fold'], y=fold_data['F1'], name='F1', mode='lines+markers', line=dict(color='#EF9F27')))
fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Random baseline")
fig.update_layout(yaxis_range=[0.35, 0.65], xaxis_title="Backtest Evaluation Splits", yaxis_title="Score Metric Value", height=380)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── FEATURE IMPORTANCE GRAPH ────────────────────────────────────────────────
st.subheader("Top 10 Live Feature Importances (From XGBoost Engine)")

importance_df = pd.DataFrame({
    'feature': ALL_FEATURES,
    'importance': importances
}).sort_values('importance', ascending=False).head(10)

fig2 = px.bar(importance_df.sort_values('importance'),
              x='importance', y='feature', orientation='h',
              color_discrete_sequence=['#378ADD'])
fig2.update_layout(height=380, xaxis_title="Relative Weight Gained", yaxis_title="")
st.plotly_chart(fig2, use_container_width=True)
