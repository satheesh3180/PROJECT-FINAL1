#PROJECT FINAL1
 Advanced Time Series Forecasting with Transformer Attention

(PROJECT FINAL1)

# Project Overview

This project implements an advanced multivariate time series forecasting system using a custom Transformer-based encoder–decoder architecture with self-attention, built from scratch in PyTorch.

The objective is to compare deep learning–based forecasting against a classical statistical baseline (SARIMAX) and to interpret attention weights to understand feature importance over time.

This project strictly follows academic requirements for:

Deep Learning

Time Series Analysis

Attention Mechanisms

Model Interpretability

Hyperparameter Optimization

# Key Objectives

Generate a complex multivariate time series dataset with correlated features

Implement a custom encoder–decoder Transformer

Handle sequence padding and masking

Perform hyperparameter optimization using Grid Search

Evaluate using MAE, RMSE, SMAPE

Compare results with SARIMAX baseline

Interpret learned attention weights

# Dataset Description

The dataset is programmatically generated to simulate realistic temporal behavior such as:

Feature Name	Description
target	Primary value to forecast
seasonality	Daily cyclical pattern
temperature	Long-term seasonal effect
industrial_load	Industry-related demand
volatility	Financial-style cumulative noise

✔ Non-stationary
✔ Cyclical
✔ Multivariate (5 features)

# Model Architecture
 #Transformer Encoder–Decoder

Encoder:

Multi-head self-attention

Feed-forward network

Residual connections & Layer Normalization

Decoder:

Attention over encoder outputs

Final dense layer for forecasting

# Key Design Choices

Built using low-level PyTorch modules

No high-level forecasting wrappers

Fully modular & PEP-8 compliant

Attention weights extracted for interpretation

# Padding & Masking

Variable-length sequences are supported using:

Zero-padding

Key padding masks in attention layers

This ensures:

Correct handling of missing timesteps

No attention leakage into padded values

# Hyperparameter Optimization

A Grid Search strategy is used to tune:

Parameter	Values
Learning Rate	0.001, 0.0005
Sequence Length	24, 48
Attention Heads	2, 4
✔ Best Configuration (Example)
Learning Rate: 0.0005
Sequence Length: 48
Attention Heads: 4

# Evaluation Metrics

The following metrics are used for rigorous evaluation:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

SMAPE (Symmetric Mean Absolute Percentage Error)

# Baseline Model (Classical)

To validate the effectiveness of deep learning, the model is compared against:

SARIMAX (Seasonal ARIMA with Exogenous Variables)
Implemented using the statsmodels library.

This provides a fair and interpretable benchmark.

# Results Summary
Model	MAE	RMSE
Transformer + Attention	 Lower	 Lower
SARIMAX	Higher	Higher

# The Transformer model consistently outperforms SARIMAX by capturing long-range dependencies and multivariate interactions.

# Attention Weight Interpretation

Key insights from attention analysis:

Seasonal patterns dominate short-term predictions

Volatility becomes important during abrupt changes

Attention dynamically shifts based on temporal context

# This improves model transparency, which is often missing in deep learning models.

# Technologies Used

Python 3.x

PyTorch

NumPy

Pandas

Scikit-learn

Statsmodels

📂 Project Structure
PROJECT FINAL1/
│
├── data_generation.py
├── transformer_model.py
├── training_and_gridsearch.py
├── evaluation.py
├── baseline_sarimax.py
└── README.md

# Conclusion

This project demonstrates that Transformer-based attention models provide:

Superior forecasting accuracy

Better handling of multivariate dependencies

Interpretability via attention weights

Compared to classical models like SARIMAX, deep learning approaches show significant performance improvements in complex forecasting scenarios.

