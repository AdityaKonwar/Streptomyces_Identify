import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
from skimage.feature import local_binary_pattern
import datetime

# ==== PARAMETERS ====
IMG_SIZE = (512, 512)
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
MODEL_PATH = "/Users/adityanarayankonwar/Library/CloudStorage/OneDrive-Personal/streptomyces_model_3_yolo11/streptomyces_model_yolo11_best.pt"

# ==== LOAD MODEL ====
model = YOLO(MODEL_PATH)

# ==== IMAGE PROCESSING FUNCTIONS ====
def generate_lbp(gray_img):
    lbp = local_binary_pattern(gray_img, LBP_POINTS, LBP_RADIUS, method='uniform')
    lbp = ((lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255).astype(np.uint8)
    return lbp

def enhance_saturation_clahe(bgr_img):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[..., 1] = clahe.apply(hsv[..., 1])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = generate_lbp(gray)
    clahe_img = enhance_saturation_clahe(img)
    hsv = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2HSV)
    sat_clahe = hsv[..., 1]
    processed = cv2.merge([lbp, sat_clahe, sat_clahe]).astype(np.uint8)
    return processed, img

# ==== STREAMLIT UI ====
st.title("🧫 Streptomyces Classifier with YOLOv11 (LBP + CLAHE Only)")

uploaded_file = st.file_uploader("Upload an agar plate image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("🔬 Preprocessing image..."):
        processed_img, original_img = preprocess_image(uploaded_file)

    with st.spinner("🤖 Predicting with YOLOv11..."):
        results = model.predict(source=processed_img, imgsz=IMG_SIZE[0])
        pred_class = results[0].probs.top1
        conf = results[0].probs.top1conf
        label = model.names[pred_class] if model.names else f"Class-{pred_class}"

    # ==== Display Results ====
    st.image(original_img[..., ::-1], caption=f"Original Image", use_column_width=True)
    st.image(processed_img[..., ::-1], caption="Preprocessed Input (LBP + CLAHE Only)", use_column_width=True)
    st.success(f"🧠 Prediction: **{label}** \nConfidence: **{conf:.2f}**")

    # ==== Optional Logging ====
    log_file = "prediction_log.csv"
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("Timestamp,ImageName,Prediction,Confidence\n")

    with open(log_file, 'a') as f:
        f.write(f"{datetime.datetime.now()},{uploaded_file.name},{label},{conf:.2f}\n")

    st.download_button("Download Prediction Log", data=open(log_file).read(), file_name="prediction_log.csv")
