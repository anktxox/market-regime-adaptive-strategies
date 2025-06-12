# market-regime-adaptive-strategies

 Finalist Submission for Inter IIT Tech Meet 13.0  
🏦 Developed for Zelta Labs as part of a portfolio optimization and trading signal generation challenge.

## 📈 Project Overview

This repository showcases a series of innovative, risk-adjusted trading strategies for BTC and ETH, including:

- 🟢 *Burning Phoenix Strategy* (Candlestick-based BTC Reversals)
- 🔵 *Technical Indicator Fusion* (RSI, Aroon, PSAR, ADX)
- 🧠 *Temporal Fusion Transformer* (Deep learning-based ETH signal prediction)
- 🧪 *Kalman Filter Smoothers* (Noise reduction + momentum signals)
- 🧩 *Portfolio Manager* (Dynamic alpha selector using Sharpe Ratio, MDD, Volatility)

## 📊 Performance Highlights

| Strategy            | Sharpe Ratio | Annualized Return | Max Drawdown |
|---------------------|--------------|--------------------|---------------|
| Master Alpha (ETH)  | 8.93         | 893.84%            | 8.49%         |
| Master Alpha (BTC)  | 6.59         | 169%               | 10.00%        |

> ✅ ETH strategy beat the benchmark 14/16 quarters and all 4 years.

## 🧠 Strategy Architecture

All strategies generate alpha signals based on real-time indicators. A portfolio manager dynamically switches between strategies weekly based on:

- 📉 Volatility
- 📊 Sharpe Ratio
- 📈 Mean Positive Returns
- 🛑 Max Drawdown

## 🧪 Technologies Used

- Python (pandas, numpy, scipy, ta-lib)
- JupyterLab
- Plotly / Matplotlib
- PyTorch (for Temporal Fusion Transformer)
- Kalman Filter, RSI, MACD, STC, ADX, ATR, Heiken Ashi

## 📂 File Structure
