import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Configuration ---
MODEL_WEIGHTS_PATH = 'authenticator_v3_pro.h5' # The path to your best model file
IMAGE_PATH = r"C:\Users\vijay\OneDrive\Pictures\WhatsApp Image 2023-07-28 at 18.13.47.jpg"   # <-- CHANGE THIS to the name of your test image
IMG_SIZE = 224                           # The input size for the MobileNetV2 model

# --- This function contains the definitive fix ---
def create_and_load_model(weights_path):
    """
    Creates a clean model architecture and loads the learned weights into it.
    This bypasses any errors with custom or non-standard layers.
    """
    # 1. Define the model architecture EXACTLY as before, but WITHOUT the preprocessing layer.
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False # It's good practice to set this

    # Build the full architecture
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # NOTE: The preprocess_input and data_augmentation layers are NOT here
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs, outputs)

    # 2. Load ONLY the weights from your saved file into this clean architecture.
    print(f"Loading weights from: {weights_path}")
    model.load_weights(weights_path)
    
    return model

# --- Main Prediction Logic ---
print("--- Starting AI Authenticity Check ---")

if not os.path.exists(MODEL_WEIGHTS_PATH):
    print(f"🔴 Error: Model file '{MODEL_WEIGHTS_PATH}' not found.")
    exit()
if not os.path.exists(IMAGE_PATH):
    print(f"🔴 Error: Image file '{IMAGE_PATH}' not found.")
    exit()

try:
    # Create the model and load the weights
    model = create_and_load_model(MODEL_WEIGHTS_PATH)

    # Load and prepare the image
    print(f"Loading and preprocessing image: {IMAGE_PATH}")
    img = Image.open(IMAGE_PATH).resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Manually apply the necessary preprocessing for MobileNetV2
    preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)

    # Make the prediction on the correctly preprocessed image
    print("AI is analyzing the image...")
    prediction = model.predict(preprocessed_img)
    score = prediction[0][0]

    # --- Interpret and Display Result ---
    print("\n--- PREDICTION RESULT ---")
    if score < 0.5:
        confidence = (1 - score) * 100
        print(f"✅ Verdict: AI-GENERATED")
        print(f"   Confidence: {confidence:.2f}%")
    else:
        confidence = score * 100
        print(f"✅ Verdict: REAL PHOTOGRAPH")
        print(f"   Confidence: {confidence:.2f}%")

except Exception as e:
    print(f"An error occurred: {e}")