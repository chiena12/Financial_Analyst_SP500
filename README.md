# Financial_Analyst_SP500

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://financialanalystsp500-ldw3wqfzxm6e6mqm3abxdh.streamlit.app/)

📈 S&P 500 Portfolio Optimization & Trend Prediction
This project integrates Modern Portfolio Theory (MPT) with Deep Learning to build an automated system for asset allocation and market trend forecasting using S&P 500 historical data.

🌟 Key Features
📊 Portfolio Optimization (Monte Carlo): 
* Simulated 20,000 random portfolios to determine the optimal asset weights for five selected stocks (ACGL, FIS, FITB, IEX, LOW).

* Identified the Maximum Sharpe Ratio for high-efficiency returns and the Minimum Volatility for risk-averse strategies.

* Generated the Efficient Frontier plot to visualize the risk-reward spectrum.

🤖 Advanced Trend Forecasting:

* Bi-LSTM (Deep Learning): A bidirectional recurrent neural network designed to capture long-term dependencies and price reversals.

* XGBoost (Machine Learning): A gradient boosting classifier used to predict next-day price direction (Bullish/Bearish).

⚙️ Quantitative Engineering: 



* Engineered technical indicators such as RSI, EMA Cross, Log Returns, and Rolling Volatility to feed the predictive models.

📈 Analysis Results
* Strategy Performance: The Bi-LSTM and XGBoost models were evaluated against a Buy & Hold benchmark to measure alpha generation.

* Risk Metrics: Integrated Sharpe Ratio and Maximum Drawdown (MDD) calculations to provide a professional assessment of strategy robustness.

* Interactive UI: Users can view daily log returns, cumulative returns, and feature importance rankings directly on the web app.
