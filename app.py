# app.py - Lung Nodule Classifier + Similar Case Retrieval
# Model loaded from Google Drive

import streamlit as st
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import faiss
import sqlite3
import numpy as np
import cv2
from PIL import Image
import time
import os
import gdown   # pip install gdown

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
GDRIVE_MODEL_URL = "https://drive.google.com/file/d/13XYtH4-6nkIqoI6Bfm97F6saItHBmkq8/view?usp=drive_link" 
LOCAL_MODEL_PATH = "temp_vit_model.pth"   # temporary file in app folder

DB_PATH         = "Database/lung_nodule.db"
INDEX_PATH      = "Database/faiss_index.index"

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K           = 5
CLASS_NAMES     = ["Normal", "Benign", "Malignant"]

st.set_page_config(page_title="Lung Nodule Classifier", layout="wide")

# Custom CSS (same as before)
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: white; }
    .stButton>button { background: linear-gradient(90deg, #667eea, #764ba2); color: white; border: none; border-radius: 12px; padding: 0.7rem 1.5rem; font-weight: bold; transition: all 0.3s; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102,126,234,0.5); }
    .card { background: rgba(255,255,255,0.08); border-radius: 16px; padding: 1.5rem; margin: 1rem 0; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.18); animation: fadeIn 0.8s ease-out; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    h1, h2, h3 { color: #a78bfa !important; }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# LOAD MODEL FROM GOOGLE DRIVE (cached)
# ────────────────────────────────────────────────
@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(LOCAL_MODEL_PATH):
        with st.spinner("Downloading model from Google Drive (first time only)..."):
            gdown.download(GDRIVE_MODEL_URL, LOCAL_MODEL_PATH, quiet=False)

    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(LOCAL_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model_from_drive()
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Load FAISS & DB
index = faiss.read_index(INDEX_PATH)
conn = sqlite3.connect(DB_PATH)

# Sidebar (same as before)
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/lungs.png", width=80)
    st.title("Lung Nodule AI")
    st.markdown("**Early Detection & Classification**")
    st.markdown("ViT-base • 96.25% 10-fold acc")
    st.markdown("---")
    st.info("Upload a lung CT patch (PNG/JPG) → get prediction + similar cases")
    st.markdown("---")
    st.caption("Built with ❤️ by Utkarsh • NIT Manipur")

# Main layout
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Upload Lung Patch")
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Patch", use_column_width=True)

        with st.spinner("Analyzing nodule..."):
            time.sleep(0.8)

            img_array = np.array(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            inputs = processor(images=img_array, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_class = np.argmax(probs)
                confidence = probs[pred_class]

                vit_outputs = model.vit(**inputs)
                feat = vit_outputs.last_hidden_state[:, 0, :].cpu().numpy()

        with st.container():
            st.markdown(f'<div class="card"><h3>Prediction</h3><h2>{CLASS_NAMES[pred_class]}</h2>'
                        f'<p>Confidence: <strong>{confidence:.2%}</strong></p></div>', 
                        unsafe_allow_html=True)

        st.bar_chart({
            "Normal": probs[0],
            "Benign": probs[1],
            "Malignant": probs[2]
        }, use_container_width=True)

with col2:
    if uploaded_file is not None:
        st.subheader("Top Similar Cases")

        with st.spinner("Searching similar nodules..."):
            distances, indices = index.search(feat.astype('float32'), TOP_K)

            cols = st.columns(5)
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:
                    continue

                cursor = conn.cursor()
                cursor.execute("SELECT path, label FROM nodules WHERE rowid = ?", (idx + 1,))
                result = cursor.fetchone()
                if result:
                    sim_path, sim_label = result
                    sim_img = Image.open(sim_path)
                    with cols[i % 5]:
                        st.image(sim_img, 
                                 caption=f"Rank {i+1}\n{CLASS_NAMES[sim_label]}\nDist: {dist:.4f}",
                                 width=150)

conn.close()

st.markdown("---")
st.caption("Model trained on LUNA16 • 10-fold CV • 96.25% mean accuracy • For research/educational use only")
