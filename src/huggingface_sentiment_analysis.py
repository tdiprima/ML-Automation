# This script uses Hugging Face's pipeline for sentiment analysis on text.
# It loads a pre-trained model and performs a prediction.

from transformers import pipeline

# Load the pipeline
classifier = pipeline("sentiment-analysis")

# Perform prediction
result = classifier(
    "This article on Python automation is absolutely brilliant and game-changing."
)

# Print result
print(result)

# Label: POSITIVE
# Score: 0.9999 (99.99% confidence)
