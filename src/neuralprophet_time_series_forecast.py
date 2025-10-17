# This script uses NeuralProphet to forecast time series data.
# It generates a sample 'time_series_data.csv' if not present, fits the model, and predicts future values.

import pandas as pd
from neuralprophet import NeuralProphet
from pathlib import Path

# Define path to CSV file
csv_path = Path(__file__).parent / "../data/time_series_data.csv"
csv_path = csv_path.resolve()

# Generate sample data if 'time_series_data.csv' doesn't exist (daily data example)
if not csv_path.exists():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    sample_data = pd.DataFrame(
        {
            "ds": dates,
            "y": [
                i + (i % 7) * 10 for i in range(100)
            ],  # Simple trend with weekly seasonality
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    sample_data.to_csv(csv_path, index=False)

# Load data
data = pd.read_csv(csv_path)

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
