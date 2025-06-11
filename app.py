import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import base64
import fitz  # PyMuPDF
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Authenticity Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Set Background Image ---
@st.cache_data
def get_base64_image(image_file):
    with open(image_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg_image(image_path):
    img_base64 = get_base64_image(image_path)
    background_css = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .block-container {{
        background-color: rgba(0, 0, 0, 0.6);
        padding: 2rem;
        border-radius: 20px;
    }}
    h1, h2, h3, p, label {{
        color: white;
    }}
    </style>
    '''
    st.markdown(background_css, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_prediction_model(weights_path):
    IMG_SIZE = 224
    base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs, outputs)
    if not os.path.exists(weights_path):
        return None
    model.load_weights(weights_path)
    return model

# --- PREDICTION ---
def predict_image(model, image_to_predict):
    IMG_SIZE = 224
    img = image_to_predict.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    img_batch = np.expand_dims(img_array, axis=0)
    preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
    prediction = model.predict(preprocessed_img)
    return prediction[0][0]

# --- MAIN APP ---
MODEL_WEIGHTS_PATH = "authenticator_v3_pro.h5"
BACKGROUND_IMAGE_PATH = "image.png"

set_bg_image(BACKGROUND_IMAGE_PATH)

st.title("🛡️ AI Authenticity Detector")
st.markdown("<p style='text-align:center;'>Is it Human... or Heuristic? Verifying Digital Origins.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload an image or a PDF to verify its authenticity.",
    type=["jpg", "jpeg", "png", "pdf"]
)

if uploaded_file is not None:
    model = load_prediction_model(MODEL_WEIGHTS_PATH)

    if model is None:
        st.error("🔴 Critical Error: The AI model file could not be loaded.")
    else:
        if uploaded_file.type == "application/pdf":
            st.info("PDF detected. Extracting and analyzing all embedded images...")
            results = []
            try:
                pdf_bytes = uploaded_file.getvalue()
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    image_list = page.get_images(full=True)

                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image = Image.open(io.BytesIO(image_bytes))

                        score = predict_image(model, image)
                        results.append({"page": page_num + 1, "image": image, "score": score})
            except Exception as e:
                st.error(f"Error processing PDF file: {e}")

            if results:
                st.markdown(f"### Analysis Complete: Found {len(results)} image(s) in the PDF.")
                real_count = sum(1 for r in results if r["score"] >= 0.5)
                fake_count = len(results) - real_count

                st.metric("Real Photographs Found", real_count)
                st.metric("AI-Generated Images Found", fake_count)

                with st.expander("Show Detailed Report"):
                    for res in results:
                        st.write(f"--- Page {res['page']} ---")
                        st.image(res['image'], width=200)
                        confidence = res['score'] * 100 if res['score'] >= 0.5 else (1 - res['score']) * 100
                        verdict = "REAL PHOTOGRAPH" if res['score'] >= 0.5 else "AI-GENERATED"
                        if res['score'] < 0.5:
                            st.error(f"Verdict: {verdict} (Confidence: {confidence:.1f}%)")
                        else:
                            st.success(f"Verdict: {verdict} (Confidence: {confidence:.1f}%)")
            else:
                st.warning("No images could be found in this PDF document.")
        else:
            image_to_process = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image_to_process, caption='Uploaded Image', use_column_width=True)
            with col2:
                with st.spinner('AI is analyzing the image...'):
                    score = predict_image(model, image_to_process)
                st.markdown("### Verification Result")
                if score < 0.5:
                    confidence = (1 - score) * 100
                    st.metric(label="Verdict", value="AI-GENERATED", delta=f"-{confidence:.1f}% Certainty", delta_color="inverse")
                    st.error("Our model has identified signs of AI generation.")
                else:
                    confidence = score * 100
                    st.metric(label="Verdict", value="REAL PHOTOGRAPH", delta=f"{confidence:.1f}% Certainty")
                    st.success("Our model has identified this as a genuine photograph.")
