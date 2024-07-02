XGBoost Regression Model for Sales Prediction

Overview

This repository contains a machine learning model built using XGBoost (Extreme Gradient Boosting) to predict sales based on provided features. The model is trained on historical data and can be used to make predictions on new datasets.

Files

train.py: Python script that trains the XGBoost regression model using historical sales data (big_mart_data.csv).
xgb_regressor_model.pkl: Pickle file containing the trained XGBoost regressor model.
predict.py: Script to load the trained model and make predictions on new data (new_data.csv).
predictions.csv: CSV file containing predicted sales values for the new data.

Dependencies

Python 3.x
numpy
pandas
matplotlib
seaborn
sklearn
xgboost
joblib

Notes

Ensure all dependencies are installed (pip install pandas scikit-learn xgboost joblib).
Modify file paths and names according to your specific dataset and environment.
For detailed implementation, refer to train.py and predict.py script in the notebook.