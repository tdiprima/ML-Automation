# Description: This script uses Optuna to optimize hyperparameters for an SVM classifier on the Iris dataset.
# It defines an objective function and runs optimization for 100 trials.

import optuna
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.datasets import load_iris

def objective(trial):
    X, y = load_iris(return_X_y=True)
    
    # Suggest hyperparameters
    C = trial.suggest_float('C', 1e-4, 1e-2, log=True)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
    
    classifier = SVC(C=C, kernel=kernel, gamma='auto')
    return cross_val_score(classifier, X, y, n_jobs=-1, cv=3).mean()

# Create and optimize study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(f"Best trial value: {study.best_value}")
print(f"Best params: {study.best_params}")
