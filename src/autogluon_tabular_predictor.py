# This script uses AutoGluon to automatically train, tune, and ensemble models on tabular data for prediction.
# It generates a sample 'data.csv' if not present, trains a predictor, and prints a leaderboard.

import pandas as pd
import numpy as np
from pathlib import Path
from autogluon.tabular import TabularDataset, TabularPredictor

path = "../data/data.csv"

# Check if data.csv exists, create it only if it doesn't
csv_path = Path(path)

if not csv_path.exists():
    # Generate larger sample dataset with 1000 rows
    np.random.seed(42)
    n_samples = 1000

    sample_data = pd.DataFrame(
        {
            "feature1": np.random.randint(1, 100, n_samples),
            "feature2": np.random.randint(10, 500, n_samples),
            "feature3": np.random.randn(n_samples),
            "feature4": np.random.uniform(0, 1, n_samples),
            "label": np.random.randint(0, 2, n_samples),
        }
    )
    sample_data.to_csv(csv_path, index=False)

# Load data
train_data = TabularDataset(path)

# Train the predictor
predictor = TabularPredictor(label="label").fit(train_data)

# Print leaderboard
print(predictor.leaderboard(train_data))

# We're looking at a ranking of models by their test scores,
# showing which algorithms performed best.
