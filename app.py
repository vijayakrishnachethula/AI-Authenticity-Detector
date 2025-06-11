import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import base64
import fitz  # The PyMuPDF library
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Authenticity Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- All helper functions (get_base64, set_video_as_page_bg, load_prediction_model, predict_image) remain the same ---
# (I will omit them here for brevity, but they should be in your final script)
@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f: data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError: return None

def set_video_as_page_bg(video_file):
    video_str = get_base64_of_bin_file(video_file)
    if video_str is None:
        st.warning(f"⚠️ Video file not found. Please check the path.")
        return
    page_bg_video = f'''
    <style>
    .stApp{{background:transparent}} #video-background{{position:fixed;top:0;left:0;width:100vw;height:100vh;object-fit:cover;z-index:-2}} #video-overlay{{position:fixed;top:0;left:0;width:100vw;height:100vh;background-color:rgba(0,0,0,.65);z-index:-1}} [data-testid="stVerticalBlock"]>[style*="flex-direction:column"]>[data-testid="stVerticalBlock"]{{background:rgba(40,40,60,.5);border-radius:20px;padding:25px;border:1px solid rgba(255,255,255,.1);backdrop-filter:blur(15px);-webkit-backdrop-filter:blur(15px)}} h1,h2,h3,p,.st-emotion-cache-1jicfl2{{color:#fff}}
    </style>
    <div id="video-overlay"></div><video id="video-background" autoplay loop muted><source src="data:video/mp4;base64,{video_str}" type="video/mp4"></video>
    '''
    st.markdown(page_bg_video, unsafe_allow_html=True)

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
    if not os.path.exists(weights_path): return None
    model.load_weights(weights_path)
    return model

def predict_image(model, image_to_predict):
    IMG_SIZE = 224
    img = image_to_predict.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    if img_array.shape[2] == 4: img_array = img_array[:, :, :3]
    img_batch = np.expand_dims(img_array, axis=0)
    preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
    prediction = model.predict(preprocessed_img)
    return prediction[0][0]

# --- APP LAYOUT ---
MODEL_WEIGHTS_PATH = "authenticator_v3_pro.h5"
VIDEO_FILE_PATH = "background.mp4.mp4"
set_video_as_page_bg(VIDEO_FILE_PATH)
model = load_prediction_model(MODEL_WEIGHTS_PATH)

if model is None:
    st.error("🔴 Critical Error: The AI model file could not be loaded.")
else:
    with st.container():
        st.title("AI Authenticity Detector")
        st.markdown("<p style='text-align: center; color: #DDDDDD;'>Is it Human... or Heuristic? Verifying Digital Origins.</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload an image or a PDF to verify its authenticity.",
            type=["jpg", "jpeg", "png", "pdf"]
        )

    if uploaded_file is not None:
        # === NEW LOGIC TO HANDLE PDFS THOROUGHLY ===
        if uploaded_file.type == "application/pdf":
            st.info("PDF detected. Extracting and analyzing all embedded images...")
            results = []
            try:
                pdf_bytes = uploaded_file.getvalue()
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                
                # Loop through every page
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    image_list = page.get_images(full=True)
                    
                    # Loop through every image on the page
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image = Image.open(io.BytesIO(image_bytes))
                        
                        # Get prediction for this specific image
                        score = predict_image(model, image)
                        results.append({"page": page_num + 1, "image": image, "score": score})
            except Exception as e:
                st.error(f"Error processing PDF file: {e}")

            # --- Display the summary report ---
            if results:
                st.markdown(f"### Analysis Complete: Found {len(results)} images in the PDF.")
                real_count = sum(1 for r in results if r["score"] >= 0.5)
                fake_count = len(results) - real_count
                
                st.metric(label="Real Photographs Found", value=real_count)
                st.metric(label="AI-Generated Images Found", value=fake_count)

                with st.expander("Show Detailed Report"):
                    for res in results:
                        st.write(f"--- Page {res['page']} ---")
                        st.image(res['image'], width=200)
                        if res['score'] < 0.5:
                            confidence = (1 - res['score']) * 100
                            st.error(f"Verdict: AI-GENERATED (Confidence: {confidence:.1f}%)")
                        else:
                            confidence = res['score'] * 100
                            st.success(f"Verdict: REAL PHOTOGRAPH (Confidence: {confidence:.1f}%)")
            else:
                st.warning("No images could be found within this PDF document.")

        else:
            # --- This is the original logic for single image uploads ---
            image_to_process = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 1], gap="large")
            with col1:
                st.image(image_to_process, caption='Image for Authentication')
            with col2:
                with st.spinner('AI is analyzing the image...'):
                    score = predict_image(model, image_to_process)
                st.markdown("### Verification Result")
                if score < 0.5:
                    confidence = (1 - score) * 100
                    st.metric(label="Verdict", value="AI-GENERATED", delta=f"-{confidence:.1f}% Certainty", delta_color="inverse")
                    st.error("Our model has identified signatures consistent with AI generation.")
                else:
                    confidence = score * 100
                    st.metric(label="Verdict", value="REAL PHOTOGRAPH", delta=f"{confidence:.1f}% Certainty")
                    st.success("Our model has identified this as a genuine photograph.")