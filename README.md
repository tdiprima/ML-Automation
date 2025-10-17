# One-Line ML Automation

This repository contains Python scripts demonstrating 7 libraries that simplify machine learning workflows to near-single-line commands, as inspired by [an article](https://medium.com/python-in-plain-english/the-7-python-libraries-that-turn-your-model-training-into-a-single-line-of-code-e2c6bab56a4c) on automating model training.

## Scripts
- **autogluon_tabular_predictor.py**: Auto-trains and ensembles models on tabular data.
- **fastai_image_classifier.py**: Fine-tunes a vision model for image classification.
- **optuna_hyperparam_tuning.py**: Optimizes hyperparameters for an SVM on Iris data.
- **pycaret_model_comparison.py**: Compares and tunes classification models.
- **mlflow_experiment_tracking.py**: Tracks ML experiments with logging.
- **huggingface_sentiment_analysis.py**: Performs sentiment analysis using transformers.
- **neuralprophet_time_series_forecast.py**: Forecasts time series data.

## Usage
Install dependencies with `uv sync`. Run each script individually. Some require internet (e.g., FastAI for data download).

## Requirements
Python 3.11. Libraries: autogluon, pandas, fastai, optuna, scikit-learn, pycaret, mlflow, transformers, neuralprophet.

## Image Credit

`cat.jpg` [Image Credit](https://pixabay.com/users/mabelamber-1377835/) **via** [Image Credit](https://files.realpython.com/media/cat.8ce1dab25b77.jpg) resized by me to 512 px.
