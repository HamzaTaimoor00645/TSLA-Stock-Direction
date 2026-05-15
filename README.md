# TSLA Stock Direction Predictor

## Overview
ML pipeline predicting Tesla stock movement using NLP sentiment,
technical indicators, and macro data. Achieved AUC 0.533 on
unseen data via 5-fold walk-forward validation.

## Stack
XGBoost · FinBERT · Alpaca API · scikit-learn · Streamlit

## Results
| Fold | Accuracy | F1    | AUC   |
|------|----------|-------|-------|
| Avg  | 52.1%    | 0.499 | 0.533 |

## Features (35 total)
- 27 technical indicators (RSI, MACD, Bollinger Bands...)
- 4 macro features (VIX, SPY returns)
- 4 FinBERT sentiment features from live news

## Methodology
Walk-forward validation — no lookahead bias...
