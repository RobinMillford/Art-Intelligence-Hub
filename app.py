import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import cv2
import time

# Load the trained model
model = load_model('MobileNetV2_finetuned_model(0.95 loss 0.11).keras')

# Define image size based on model input shape
IMAGE_SIZE = (224, 224)

def set_bg_hack_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("https://i.redd.it/jvccxng1cbzd1.png");
            background-size: cover;
            backdrop-filter: blur(10px);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def preprocess_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = image.resize(IMAGE_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def generate_gradcam_heatmap(model, img_array, last_conv_layer_name="out_relu", pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def overlay_heatmap_on_image(img, heatmap, alpha=0.4):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return superimposed_img

def classify_image_with_gradcam(image_data):
    processed_image = preprocess_image(image_data)
    prediction = model.predict(processed_image)
    class_label = int(np.round(prediction[0][0]))
    confidence = prediction[0][0]

    heatmap = generate_gradcam_heatmap(model, processed_image)
    overlay_img = overlay_heatmap_on_image(image_data, heatmap)
    return class_label, confidence, overlay_img

set_bg_hack_url()

st.markdown("<h1 style='text-align: center; font-size: 48px;'>AI vs Real Art Image Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 24px;'>Upload one or more images and the model will classify each as either AI Art or Real Art.</p>", unsafe_allow_html=True)

# Model Details Expander
with st.expander("📊 Model Details"):
    st.write("""
    This application is powered by a fine-tuned MobileNetV2 model trained on a total of 27,266 images. The model achieved a training accuracy of 94.50% and a validation accuracy of 95.88% based on 3,370 validation images. Despite these high accuracy rates, it may not yield perfectly accurate results in all cases due to variability in image characteristics. 
    
    This model leverages a transfer learning approach with MobileNetV2 to effectively distinguish between AI-generated and real art images. A Grad-CAM heatmap visualization is provided to help interpret the model's predictions by highlighting areas in the image most relevant to the classification.
    """)

# Image upload and display
uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if 'page' not in st.session_state:
        st.session_state.page = 0

    current_image_index = st.session_state.page
    num_images = len(uploaded_files)

    st.markdown("<h3 style='text-align: center; font-size: 24px;'>Processing...</h3>", unsafe_allow_html=True)

    # Load and show current image
    image = Image.open(uploaded_files[current_image_index])

    # Progress bar to simulate transition effect
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    st.success("Image uploaded successfully")

    # Make prediction with Grad-CAM visualization
    class_label, confidence, overlay_img = classify_image_with_gradcam(image)

    # Horizontal layout for displaying results
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption=f"Uploaded Image - {uploaded_files[current_image_index].name}", use_column_width=True)

    with col2:
        if class_label == 0:
            st.markdown(f"<h2 style='text-align: center; font-size: 32px; color: red;'>Classified as AI Art</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='text-align: center; font-size: 32px; color: green;'>Classified as Real Art</h2>", unsafe_allow_html=True)

        st.markdown(f"<h4 style='text-align: center; font-size: 24px;'>Prediction confidence: {confidence:.4f}</h4>", unsafe_allow_html=True)
        st.image(overlay_img, caption="Grad-CAM Visualization", use_column_width=True)

    # Pagination controls
    col1, col2, col3 = st.columns([1, 1, 1])

    if current_image_index > 0:
        if col1.button("< Previous", use_container_width=True):
            st.session_state.page -= 1

    if current_image_index < num_images - 1:
        if col3.button("Next >", use_container_width=True):
            st.session_state.page += 1

    # Custom Clear All Responses button after pagination controls
    st.markdown(
        """
        <div style="display: flex; justify-content: center; margin-top: 20px;">
            <form action="/" method="get">
                <button style="font-size: 20px; padding: 10px 20px;" type="submit">Clear Results and Upload New Images</button>
            </form>
        </div>
        """, unsafe_allow_html=True
    )

# Copyright Footer
st.markdown(
    """
    <hr>
    <div style='text-align: center;'>
        <p style='font-size: 1.2em; font-family: "Arial", sans-serif;'>
            © 2024 All rights reserved by 
            <a href='https://github.com/RobinMillford' target='_blank'>
                <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Github-desktop-logo-symbol.svg/2048px-Github-desktop-logo-symbol.svg.png' height='30' style='vertical-align: middle;'>
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)