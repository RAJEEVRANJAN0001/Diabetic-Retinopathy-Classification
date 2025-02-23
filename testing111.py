import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
MODEL_PATH = "diabetic_retinopathy_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Dataset Path
TEST_DATASET_PATH = "/Users/rajeevranjanpratapsingh/PycharmProjects/dataminig /aptos2019-blindness-detection/test_images"

# Image Parameters
IMG_SIZE = (224, 224)

# Load all images from the test folder
test_image_paths = [os.path.join(TEST_DATASET_PATH, img) for img in os.listdir(TEST_DATASET_PATH) if img.endswith(('.png', '.jpg', '.jpeg'))]

# Convert images to arrays
test_images = np.array([img_to_array(load_img(img, target_size=IMG_SIZE)) / 255.0 for img in test_image_paths])

# Predict
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
confidence_scores = np.max(predictions, axis=1)

# Prepare Data for CSV
file_names = [os.path.basename(img) for img in test_image_paths]

results_df = pd.DataFrame({
    "Filename": file_names,
    "Predicted Class": predicted_classes,
    "Confidence Score": confidence_scores
})

# Save to CSV
CSV_PATH = "predictions.csv"
results_df.to_csv(CSV_PATH, index=False)

print(f"Predictions saved to {CSV_PATH}")
