# Description: This script uses PyCaret to set up data, compare classification models, and tune the best one.
# It generates a sample 'your_dataset.csv' if not present.

import pandas as pd
from pycaret.classification import *

# Generate sample data if 'your_dataset.csv' doesn't exist (for completeness)
sample_data = pd.DataFrame(
    {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "target_column": ["A", "B", "A", "B", "A"],
    }
)
sample_data.to_csv("your_dataset.csv", index=False)

# Load data
data = pd.read_csv("your_dataset.csv")

# Setup pipeline
setup(
    data, target="target_column", silent=True, html=False
)  # Added html=False for non-interactive mode

# Compare models
best_model = compare_models()

# Tune the best model
tuned_best = tune_model(best_model)

# Print results (for completeness)
print(tuned_best)
