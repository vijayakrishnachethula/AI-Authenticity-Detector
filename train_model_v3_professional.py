import tensorflow as tf
from tensorflow.keras import layers, models
import os

# --- Configuration ---
IMG_SIZE = 224 # Pre-trained models like MobileNetV2 work best with 224x224 images
BATCH_SIZE = 32
EPOCHS = 10
MODEL_SAVE_PATH = 'authenticator_v3_pro.h5'

# --- Data Loading (Unchanged) ---
train_dir = 'train'
test_dir = 'test'
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, labels='inferred', label_mode='binary',
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, shuffle=True
)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir, labels='inferred', label_mode='binary',
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, shuffle=False
)

# --- NEW: Data Augmentation Layer ---
# This layer will be part of our model to randomly transform images
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# --- NEW: Build the Model with Transfer Learning ---
print("\nBuilding model with Transfer Learning (MobileNetV2)...")

# 1. Load the pre-trained MobileNetV2 model without its top classification layer
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False, # Do not include the final ImageNet classifier layer
    weights='imagenet'  # Load weights pre-trained on ImageNet
)

# 2. Freeze the base model
# We don't want to retrain the millions of parameters in MobileNetV2
base_model.trainable = False

# 3. Create our new model on top
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# Apply data augmentation
x = data_augmentation(inputs)
# Apply MobileNetV2's preprocessing function
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
# Run the base model
x = base_model(x, training=False)
# Pool the features
x = layers.GlobalAveragePooling2D()(x)
# Add a dropout layer for regularization
x = layers.Dropout(0.2)(x)
# Add our final prediction layer
outputs = layers.Dense(1, activation='sigmoid')(x)

# Combine into the final model
model = models.Model(inputs, outputs)

# --- Compile and Train ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\n--- Starting Training for Professional Model (V3) ---")
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS
)

model.save(MODEL_SAVE_PATH)
print(f"\n✅ Professional model saved as '{MODEL_SAVE_PATH}'")