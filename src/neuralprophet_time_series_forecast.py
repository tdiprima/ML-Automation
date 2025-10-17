# Description: This script uses NeuralProphet to forecast time series data.
# It generates a sample 'time_series_data.csv' if not present, fits the model, and predicts future values.

import pandas as pd
from neuralprophet import NeuralProphet

# Generate sample data if 'time_series_data.csv' doesn't exist (for completeness; daily data example)
dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
sample_data = pd.DataFrame(
    {
        "ds": dates,
        "y": [
            i + (i % 7) * 10 for i in range(100)
        ],  # Simple trend with weekly seasonality
    }
)
sample_data.to_csv("time_series_data.csv", index=False)

# Load data
data = pd.read_csv("time_series_data.csv")

# Initialize and fit model
m = NeuralProphet()
metrics = m.fit(data, freq="D")

# Make future dataframe and predict
future = m.make_future_dataframe(data, periods=30)
forecast = m.predict(future)

# Print tail of forecast
print(forecast[["ds", "yhat1"]].tail())
