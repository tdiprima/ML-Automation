# This script uses PyCaret to set up data, compare classification models, and tune the best one.
# It generates a sample 'your_dataset.csv' if not present.

import pandas as pd
from pycaret.classification import *
import numpy as np
from pathlib import Path

# Define path to CSV file
csv_path = Path(__file__).parent / "../data/your_dataset.csv"
csv_path = csv_path.resolve()

# Generate sample data if 'your_dataset.csv' doesn't exist
if not csv_path.exists():
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
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    sample_data.to_csv(csv_path, index=False)

# Load data
data = pd.read_csv(csv_path)

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
