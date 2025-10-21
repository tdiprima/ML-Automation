"""
This script uses AutoGluon to automatically train, tune, and ensemble models
on tabular data for prediction. It trains a predictor and prints a leaderboard.
"""

from autogluon.tabular import TabularDataset, TabularPredictor

path = "../data/data.csv"

# Load data
train_data = TabularDataset(path)

# Train the predictor
predictor = TabularPredictor(label="label").fit(train_data)

# Print leaderboard
print(predictor.leaderboard(train_data))

# We're looking at a ranking of models by their test scores,
# showing which algorithms performed best.
