# Forecasting Sticker Sales with Time Series Analysis and Machine Learning

## Overview

In this project, we'll apply time series analysis and machine learning techniques to forecast sticker sales for a hypothetical online store. We'll use historical data on daily sales to predict future sales and explore the impact of various factors such as seasonality, trends, and external events.

## Data

The dataset used in this project is a sample of daily sticker sales data from January 1, 2010 to December 31, 2016. The data includes the following features:

* `date`: Date of sale
* `country`: Country of sale
* `store`: Store of sale
* `product`: Product of sale
* `num_sold`: Number of stickers sold on that day

## Approach

We will use a structured pipeline to forecast sticker sales based on historical data. The process follows these steps:

1. Data Preprocessing 
    * Load the training and test datasets. 
    * Handle missing values and outliers using IQR filtering. 
    * Convert date features into useful time-based components (e.g., year, month, day of the week). 
    * Encode categorical variables properly.
2. Feature Engineering 
   * Create Lag Features to capture past sales trends. 
   * Implement Rolling Statistics (mean, standard deviation) for smoothing and trend detection. 
   * Use Seasonal Decomposition to separate trend, seasonal, and residual components. 
   * Incorporate External Factors such as GDP per capita for economic impact.
3. Time Series Analysis
   * Analyze sales trends over time using decomposition. 
   * Identify seasonal patterns and demand fluctuations. 
   * Check for stationarity and trends in the dataset. 
4. Model Selection 
   * Train and evaluate different machine learning models such as:
   * ARIMA/SARIMA (for classical forecasting). 
   * XGBoost (tree-based regression). 
   * LSTMs (for deep learning-based forecasting). 
   * Use cross-validation with TimeSeriesSplit to ensure robustness.
   Hyperparameter Optimization 
5. Implement Optuna to find the best hyperparameters for the chosen model.
6. Model Training & Evaluation 
   * Train the final model using the best hyperparameters. 
   * Evaluate model performance using metrics such as MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error). 
   * Save the trained model for future predictions. 
7. Prediction and Submission 
   * Apply the trained model to the test dataset. 
   * Convert log-transformed predictions back to the original scale. 
   * Save the final results for submission.

## Files

* `data`: Sample dataset of daily sticker sales data divided in train/test sets.
* `main.py`: Code for time series analysis, feature engineering, and model selection.
* `final_model.pkl`: Trained XGBoost model saved as a pickle file.
* `submission.csv`: Output file containing predicted future sales of test data.

## Requirements

To run this project, install the following dependencies:

### Python Libraries

* Python 3.8+
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Statsmodels
* Optuna
* Pickle

### Hardware Requirements

A system with at least 8GB RAM for handling large datasets.

## Contributing

Feel free to contribute by suggesting new ideas, improving the code, or adding more features!

## License

This project is licensed under the MIT License – you are free to use and modify it.