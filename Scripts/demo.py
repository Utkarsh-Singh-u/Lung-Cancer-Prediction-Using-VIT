# demo.py
import streamlit as st
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import cv2
import numpy as np

st.title("Lung Nodule Classifier (ViT - 97.74% Accuracy)")

MODEL_PATH = "Model/vit_fold8_best.pth"  # ← change to your best fold
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

class_names = ["Normal", "Benign", "Malignant"]

uploaded_file = st.file_uploader("Upload a lung patch image (PNG/JPG)", type=["png", "jpg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Show image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs).logits
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]

    st.success(f"**Prediction: {class_names[pred_class]}**")
    st.write(f"Confidence: {confidence:.4f}")

    # Show probabilities
    st.bar_chart({
        "Normal": probs[0],
        "Benign": probs[1],
        "Malignant": probs[2]
    })