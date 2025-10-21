# This script uses Optuna to optimize hyperparameters for an SVM classifier on the Iris dataset.
# It defines an objective function and runs optimization for 100 trials.
# Pro Tip: Uses MedianPruner to stop underperforming trials early and save compute time.

import optuna
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.svm import SVC


def objective(trial):
    X, y = load_iris(return_X_y=True)

    # Suggest hyperparameters
    C = trial.suggest_float("C", 1e-4, 1e-2, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])

    classifier = SVC(C=C, kernel=kernel, gamma="auto")

    # Use manual cross-validation to report intermediate values for pruning
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for step, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        classifier.fit(X_train, y_train)
        score = classifier.score(X_val, y_val)
        scores.append(score)

        # Report intermediate value for pruning
        intermediate_value = sum(scores) / len(scores)
        trial.report(intermediate_value, step)

        # Allow pruner to stop unpromising trials
        if trial.should_prune():
            raise optuna.TrialPruned()

    return sum(scores) / len(scores)


# Create study with MedianPruner - stops trials performing worse than median
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,  # Don't prune first 5 trials (need baseline)
    n_warmup_steps=1,  # Start pruning after 1 CV fold
    interval_steps=1,  # Check for pruning after each fold
)

study = optuna.create_study(direction="maximize", pruner=pruner)
study.optimize(objective, n_trials=100)

print(f"Best trial value: {study.best_value}")
print(f"Best params: {study.best_params}")
print(
    f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
)

# Best trial value: 0.94
# Best params: {'C': 0.009556347487713545, 'kernel': 'linear'}
# Number of pruned trials: 28
