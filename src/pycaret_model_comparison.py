# This script uses PyCaret to set up data, compare classification models, and tune the best one.
# It generates a sample 'your_dataset.csv' if not present.

import pandas as pd
from pycaret.classification import *

# Generate sample data if 'your_dataset.csv' doesn't exist (for completeness)
import numpy as np

np.random.seed(42)
n_samples = 200

sample_data = pd.DataFrame(
    {
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples) * 10 + 25,
        "feature3": np.random.randint(0, 100, n_samples),
        "target_column": np.random.choice(["A", "B"], n_samples),
    }
)
sample_data.to_csv("your_dataset.csv", index=False)

# Load data
data = pd.read_csv("your_dataset.csv")

# Setup pipeline
setup(
    data, target="target_column", session_id=123, verbose=False
)  # verbose=False reduces output in PyCaret 3.0+

# Compare models
best_model = compare_models()

# Tune the best model
tuned_best = tune_model(best_model)

# Print results
print(tuned_best)

# We're looking at a benchmark of different machine learning algorithms,
# then a detailed evaluation of the winner after optimization.
