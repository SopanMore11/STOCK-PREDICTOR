# Stock Price Prediction Using LSTM

This repository contains a project that predicts the next day’s stock price for six steel sector companies using a Long Short-Term Memory (LSTM) model. The project involves end-to-end processing, from data collection and feature engineering to model training and deployment, showcasing advanced time-series analysis techniques.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Data Collection](#data-collection)
- [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Overview

The goal of this project is to predict the next day’s stock price for six steel sector companies using historical stock data. The model uses 60 days of historical data (lagging inputs) and engineered features to make predictions. This project demonstrates the application of LSTMs for time-series forecasting in the financial domain.

## Features

- Collects historical stock data using the `yfinance` library.
- Preprocesses data with robust cleaning and scaling techniques.
- Engineers advanced features, including:
  - Cyclic date features (sine and cosine transforms for day-of-week and month).
  - Financial indicators such as EMA, RSI, and MACD.
- Implements sliding window techniques to create lagging inputs.
- Trains an LSTM model optimized for time-series forecasting.
- Modular training and prediction pipeline.

## Data Collection

Historical stock data for six steel sector companies is collected from Yahoo Finance using the `yfinance` library. The data includes daily stock prices and other relevant metrics.

## Data Preprocessing and Feature Engineering

The preprocessing pipeline includes:

1. **Data Cleaning**: Handling missing values and ensuring consistent formats.
2. **Feature Scaling**: Standardization of input features to improve model performance.
3. **Feature Engineering**: Adding informative features to enrich the dataset:
   - **Cyclic Date Features**: Encoding temporal data with sine and cosine transforms.
   - **Financial Indicators**:
     - Exponential Moving Average (EMA)
     - Relative Strength Index (RSI)
     - Moving Average Convergence Divergence (MACD)
4. **Sliding Window Technique**: Creating sequences of 60 days of data as input for the LSTM model.

## Model Architecture

The LSTM model is designed to capture temporal dependencies in the stock price data. Key components include:

- Multi-layered LSTM network.
- Dropout regularization to prevent overfitting.
- Optimization using the Adam optimizer and Mean Squared Error (MSE) as the loss function.

## Training Pipeline

The training pipeline consists of the following steps:

1. Prepare data sequences using the sliding window technique.
2. Split the data into training, validation, and test sets.
3. Train the LSTM model on the training set and evaluate on the validation set.
4. Save the trained model for deployment.

## Results

The model demonstrates the ability to predict the next day’s stock price with reasonable accuracy, leveraging the temporal dependencies in the data and the engineered features.

## Technologies Used

- **Python Libraries**:
  - `yfinance` for data collection.
  - `NumPy` and `Pandas` for data manipulation.
  - `scikit-learn` for feature scaling and preprocessing.
  - `TensorFlow/Keras` for model development.

## Future Enhancements

- Incorporate sentiment analysis of financial news to improve predictions.
- Extend the model to predict multiple days ahead.
- Deploy the model as a web application using Flask or FastAPI.
- Enhance interpretability with SHAP or LIME for feature importance analysis.

## License

This project is licensed under the [MIT License](LICENSE).

---

