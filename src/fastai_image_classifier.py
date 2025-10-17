# This script uses FastAI to fine-tune a ResNet34 model for image classification (e.g., pets).
# It downloads pet images via untar_data. For custom predictions, get sample images (e.g., 'cat.jpg' or 'dog.jpg') from any source like Unsplash or your local files.

from fastai.vision.all import *

# Download and prepare data
path = untar_data(URLs.PETS) / "images"

# Define label function
dls = ImageDataLoaders.from_name_func(
    path,
    get_image_files(path),
    valid_pct=0.2,
    label_func=lambda x: "cat" if x[0].isupper() else "dog",
    item_tfms=Resize(224),
)

# Train and fine-tune the model
learn = vision_learner(dls, resnet34, metrics=error_rate).fine_tune(1)

# Example prediction
img = PILImage.create("../data/cat.jpg")

pred, pred_idx, probs = learn.predict(img)
print(f"Prediction: {pred}; Probability: {probs[pred_idx]:.4f}")

# 
# 
