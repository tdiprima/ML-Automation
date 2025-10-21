# This script uses NeuralProphet to forecast time series data.
# It fits the model and predicts future values.

import pandas as pd
from neuralprophet import NeuralProphet

# Load data (must have 'ds' for date and 'y' for value)
data = pd.read_csv("../data/time_series_data.csv")

# Initialize and fit model
m = NeuralProphet()
metrics = m.fit(data, freq="D")

# Make future dataframe and predict
future = m.make_future_dataframe(data, periods=30)
forecast = m.predict(future)

# Print tail of forecast
print(forecast[["ds", "yhat1"]].tail())

# We're watching a neural network learn patterns in time series data,
# then generate future predictions.
