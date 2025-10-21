"""
This script uses PyCaret to set up data, compare classification models,
and tune the best one.
"""

import pandas as pd
from pycaret.classification import *

# Load data
data = pd.read_csv("../data/my_dataset.csv")

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
