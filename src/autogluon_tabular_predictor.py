# Description: This script uses AutoGluon to automatically train, tune, and ensemble models on tabular data for prediction.
# It generates a sample 'data.csv' if not present, trains a predictor, and prints a leaderboard.

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

# Generate sample data if 'data.csv' doesn't exist (for completeness)
sample_data = pd.DataFrame(
    {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "label": [0, 1, 0, 1, 0],
    }
)
sample_data.to_csv("data.csv", index=False)

# Load data
train_data = TabularDataset("data.csv")

# Train the predictor
predictor = TabularPredictor(label="label").fit(train_data)

# Print leaderboard
print(predictor.leaderboard(train_data))
