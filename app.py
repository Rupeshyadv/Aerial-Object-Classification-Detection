import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from ultralytics import YOLO
import os
import requests

MOBILENETV2_URL = "https://huggingface.co/rupesh9987/aerial-classifier-models/resolve/main/mobilenetv2_aerial.keras"
YOLO_BEST_URL = "https://huggingface.co/rupesh9987/aerial-classifier-models/resolve/main/yolo_best.pt"
def download_file(url, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if not os.path.exists(filename):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

@st.cache_resource
def load_models():
    download_file(MOBILENETV2_URL, "./models/hugging_face/mobilenet.keras")
    download_file(YOLO_BEST_URL, "./models/hugging_face/yolo.pt")

    mobilenet = load_model("./models/hugging_face/mobilenet.keras")
    yolo = YOLO("./models/hugging_face/yolo.pt")

    return mobilenet, yolo

mobilenet_model, yolo_model = load_models()

CLASS_NAMES = ["bird", "drone"]

# Image Preprocessing
def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


st.set_page_config(page_title="Aerial Classifier", layout="wide")
st.title("Aerial Object Classification System")
st.write("Classify and detect whether an object is a **Bird or Drone**")

mode = st.radio("Select Mode", ["Classification", "Detection (YOLO)"])

model_choice = st.selectbox(
    "Select Model",
    ["MobileNetV2 (Recommended)"]
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Inference
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input Image", use_container_width=True)

    # Classification
    if mode == "Classification":
        processed_img = preprocess_image(image)

        if model_choice == "MobileNetV2 (Recommended)":
            prediction = mobilenet_model.predict(processed_img)

        prob = float(prediction[0][0])

        if prob > 0.5:
            label = "Drone 🚁"
            confidence = prob
        else:
            label = "Bird 🐦"
            confidence = 1 - prob

        with col2:
            st.subheader("Prediction")
            st.success(label)
            st.write(f"Confidence: {confidence:.2%}")
            st.progress(confidence)

    # Detection (YOLO)
    else:
        img_np = np.array(image)

        results = yolo_model(img_np)

        annotated_img = results[0].plot()

        with col2:
            st.subheader("Detection Output")
            st.image(annotated_img, use_container_width=True)