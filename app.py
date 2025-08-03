import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import cv2
import pandas as pd

# --- App Configuration ---
st.set_page_config(
    page_title="Art Intelligence Hub",
    page_icon="🎨",
    layout="wide"
)

# --- Model Paths ---
# Assumes models are in a 'models' subfolder relative to app.py
MODELS_DIR = 'models'
AI_VS_REAL_MODEL_PATH = os.path.join(MODELS_DIR, 'MobileNetV2_finetuned_model(0.95 loss 0.11).keras')
EFFNET_MODEL_PATH = os.path.join(MODELS_DIR, 'best_art_model_effnetv2_b2.keras')
CONVNEXT_MODEL_PATH = os.path.join(MODELS_DIR, 'best_art_model_convnext_base.keras')

# --- Class Names (Alphabetical Order for Ensemble Model) ---
STYLE_CLASS_NAMES = sorted([
    'art_nouveau', 'baroque', 'expressionism', 'impressionism', 
    'post_impressionism', 'realism', 'renaissance', 'romanticism', 
    'surrealism', 'ukiyo_e'
])
AI_REAL_CLASS_NAMES = ['AI Art', 'Real Art']

# --- Modern UI Styling (Dark Theme) ---
def load_css():
    st.markdown("""
    <style>
        
        .stApp {
            background-color: #1A202C; /* Dark charcoal background */
        }

        /* --- Main Content Container (Card UI) --- */
        .st-emotion-cache-18ni7ap, .st-emotion-cache-10trblm {
            background-color: #2D3748; /* Slightly lighter dark card */
            border-radius: 1rem;
            padding: 2.5rem;
            box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            border: 1px solid #4A5568;
        }

        /* --- Sidebar Styling --- */
        [data-testid="stSidebar"] {
            background-color: #2D3748;
            border-right: 1px solid #4A5568;
        }

        /* --- Typography --- */
        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            color: #FFFFFF;
            text-align: center;
            padding-bottom: 1rem;
        }
        
        h2, h3 {
            font-weight: 600;
            color: #E2E8F0;
        }

        /* --- Button Styling --- */
        .stButton>button {
            border-radius: 0.5rem;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
            background-image: linear-gradient(to right, #4f46e5, #818cf8);
            color: white;
            border: none;
        }
        .stButton>button:hover {
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            transform: translateY(-2px);
        }
        .stButton>button:active {
            transform: translateY(0px);
        }

        /* --- File Uploader Styling --- */
        .stFileUploader {
            border: 2px dashed #4A5568;
            border-radius: 0.75rem;
            padding: 2rem;
            background-color: #1A202C;
        }

        /* --- Custom Result Box Styling --- */
        .result-box {
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-top: 1rem;
            text-align: center;
            animation: fadeIn 0.5s ease-in-out;
        }
        .success {
            background-color: #064E3B;
            border: 1px solid #34D399;
        }
        .error {
            background-color: #7F1D1D;
            border: 1px solid #F87171;
        }
        .result-box h3 {
            margin: 0;
            font-size: 1.75rem;
            font-weight: 700;
        }
        .result-box p {
            margin-top: 0.5rem;
            font-size: 1.1rem;
            font-weight: 500;
        }
        .success h3, .success p { color: #A7F3D0; }
        .error h3, .error p { color: #FECACA; }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)

# --- Model Loading Functions ---
@st.cache_resource
def load_all_models():
    models = {'ai_vs_real': None, 'effnet': None, 'convnext': None}
    
    def _load_single_model(model_key, path, name):
        if os.path.exists(path):
            try:
                models[model_key] = tf.keras.models.load_model(path)
                st.sidebar.success(f"{name} model loaded.")
            except Exception as e:
                st.sidebar.error(f"Error loading {name} model: {e}")
        else:
            st.sidebar.error(f"{name} model not found. Expected at: '{path}'")

    _load_single_model('ai_vs_real', AI_VS_REAL_MODEL_PATH, "Artwork Authenticity")
    _load_single_model('effnet', EFFNET_MODEL_PATH, "EfficientNetV2")
    _load_single_model('convnext', CONVNEXT_MODEL_PATH, "ConvNeXt")
        
    return models

# --- Visualization Helper Functions ---
def overlay_heatmap(img, heatmap, alpha=0.5):
    if heatmap is None:
        return np.array(img)
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
    superimposed_img = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

# --- Page 1: Artwork Authenticity Classifier ---
def page_artwork_authenticity(model):
    st.header("Artwork Authenticity Classifier")
    st.markdown("Upload an image to determine if it was created by a human or generated by AI.")
    
    def preprocess_image_binary(image):
        img = image.resize((224, 224)).convert('RGB')
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        return np.expand_dims(img_array, axis=0) / 255.0

    def generate_gradcam_for_mobilenet(img_array):
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer("out_relu").output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_channel = predictions[0]
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
        return heatmap

    uploaded_file = st.file_uploader("Upload an artwork", type=["jpg", "jpeg", "png"], key="authenticity_uploader")

    if uploaded_file:
        image = Image.open(uploaded_file)
        
        with st.spinner("Analyzing..."):
            processed_image = preprocess_image_binary(image)
            prediction = model.predict(processed_image)
            class_label_index = int(np.round(prediction[0][0]))
            class_label = AI_REAL_CLASS_NAMES[class_label_index]
            confidence = (prediction[0][0] if class_label_index == 1 else 1 - prediction[0][0]) * 100
            
            heatmap = generate_gradcam_for_mobilenet(processed_image)
            overlay_img = overlay_heatmap(image, heatmap)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Artwork", use_container_width=True)
        with col2:
            st.image(overlay_img, caption="Grad-CAM: Why the model decided this", use_container_width=True)
        
        st.markdown("---")
        if class_label == "Real Art":
            st.markdown(f'<div class="result-box success"><h3>Prediction: Real Art</h3><p>Confidence: {confidence:.2f}%</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-box error"><h3>Prediction: AI Art</h3><p>Confidence: {confidence:.2f}%</p></div>', unsafe_allow_html=True)

# --- Page 2: Art Style Classifier ---
def page_art_style(effnet_model, convnext_model):
    st.header("Art Style Ensemble Classifier")
    st.markdown("This tool uses two powerful models to identify the artistic style of a painting. The final verdict is an average of both models' predictions for a more robust result.")
    
    def preprocess_for_effnet(image):
        img = image.resize((260, 260)).convert("RGB")
        img_array = np.array(img)
        return tf.keras.applications.efficientnet_v2.preprocess_input(np.expand_dims(img_array, axis=0))

    def preprocess_for_convnext(image):
        img = image.resize((224, 224)).convert("RGB")
        img_array = np.array(img)
        return tf.keras.applications.convnext.preprocess_input(np.expand_dims(img_array, axis=0))

    uploaded_file = st.file_uploader("Upload an artwork", type=["jpg", "jpeg", "png"], key="style_uploader")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Artwork", use_container_width=True)
        
        with st.spinner("Analyzing with the ensemble..."):
            effnet_input = preprocess_for_effnet(image)
            convnext_input = preprocess_for_convnext(image)
            
            pred_effnet = effnet_model.predict(effnet_input)[0]
            pred_convnext = convnext_model.predict(convnext_input)[0]
            
            ensemble_pred = (pred_effnet + pred_convnext) / 2.0
            ensemble_class_index = np.argmax(ensemble_pred)
            ensemble_confidence = np.max(ensemble_pred) * 100
            ensemble_class_name = STYLE_CLASS_NAMES[ensemble_class_index]

        st.subheader("Final Verdict (Ensemble)")
        style_name = ensemble_class_name.replace('_', ' ').title()
        st.markdown(f'<div class="result-box success"><h3>Style: {style_name}</h3><p>Confidence: {ensemble_confidence:.2f}%</p></div>', unsafe_allow_html=True)
        st.progress(int(ensemble_confidence))

        with st.expander("Show Individual Model Predictions & Reasoning"):
            
            def get_top_predictions(preds, top_k=3):
                top_indices = preds.argsort()[-top_k:][::-1]
                top_confidences = preds[top_indices]
                top_class_names = [STYLE_CLASS_NAMES[i].replace('_', ' ').title() for i in top_indices]
                df = pd.DataFrame({
                    "Confidence": top_confidences * 100,
                    "Style": top_class_names
                })
                return df

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("EfficientNetV2's Opinion")
                effnet_top_preds_df = get_top_predictions(pred_effnet)
                st.bar_chart(effnet_top_preds_df, x="Style", y="Confidence")

            with col2:
                st.subheader("ConvNeXt's Opinion")
                convnext_top_preds_df = get_top_predictions(pred_convnext)
                st.bar_chart(convnext_top_preds_df, x="Style", y="Confidence")

# --- Main App Logic ---
def main():
    load_css()
    models = load_all_models()

    st.sidebar.title("🎨 Art Intelligence Hub")
    st.sidebar.markdown("---")
    
    # --- MODIFIED (FIX): Use session state and buttons for navigation ---
    if 'page' not in st.session_state:
        st.session_state.page = "Artwork Authenticity"

    if st.sidebar.button("Artwork Authenticity", use_container_width=True):
        st.session_state.page = "Artwork Authenticity"
    if st.sidebar.button("Art Style Classifier", use_container_width=True):
        st.session_state.page = "Art Style Classifier"
    
    st.sidebar.markdown("---")
    st.sidebar.info("This application showcases different deep learning models for art analysis. Upload an image on your chosen page to begin.")

    if st.session_state.page == "Artwork Authenticity":
        if models['ai_vs_real']:
            page_artwork_authenticity(models['ai_vs_real'])
        else:
            st.error("The 'Artwork Authenticity' model is not available. Please check your `models` folder.")
            
    elif st.session_state.page == "Art Style Classifier":
        if models['effnet'] and models['convnext']:
            page_art_style(models['effnet'], models['convnext'])
        else:
            st.error("One or both of the Art Style Classifier models are not available. Please check your `models` folder.")

if __name__ == "__main__":
    main()