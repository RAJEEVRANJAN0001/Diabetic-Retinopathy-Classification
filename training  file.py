import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset Path
DATASET_PATH = "classified_images"  # Change this to your dataset path

# Image Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data Augmentation & Preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load Training & Validation Data
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation'
)

# Build DenseNet-121 Model
base_model = DenseNet121(weights='/Users/rajeevranjanpratapsingh/PycharmProjects/dataminig /densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(5, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
epochs = 40
model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# Save Model
model.save("diabetic_retinopathy_model.h5")

print("Model training complete and saved as diabetic_retinopathy_model.h5")
