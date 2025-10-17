# Description: This script uses MLFlow to track an experiment: logging params, training a RandomForest model on Iris data, and logging metrics/models.

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Added split for proper evaluation

# Initialize a new MLFlow run
with mlflow.start_run():
    n_estimators = 100
    
    # Log the parameters
    mlflow.log_param("n_estimators", n_estimators)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    # Log the metric
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log the model artifact
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    print(f"Logged accuracy: {accuracy}")
  
