# isr_setup.py
# Creates SQLite database + FAISS index from your processed patches

import os
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor
import faiss
import sqlite3
import numpy as np
import cv2
from tqdm import tqdm

# ===================== CONFIGURATION =====================
MODEL_PATH = "Model/vit_fold8_best.pth"   # ← change to your actual best fold
DATA_ROOT = "Data/processed/patches"
DB_PATH = "Database/lung_nodule.db"
INDEX_PATH = "Database/faiss_index.index"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================================================

# Load model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=3,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# We only need the ViT backbone (without classification head)
# This is the clean way — no nn.Sequential slicing needed
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Connect to SQLite
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS nodules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT UNIQUE,
        label INTEGER,
        features BLOB
    )
''')
conn.commit()

# FAISS index (L2 distance, 768-dim features)
dimension = 768
index = faiss.IndexFlatL2(dimension)

paths_list = []
labels_list = []
features_list = []

# Process all patches
classes = ["normal", "benign", "malignant"]
for cls_idx, cls_name in enumerate(classes):
    folder = os.path.join(DATA_ROOT, cls_name)
    if not os.path.exists(folder):
        print(f"Warning: Folder not found: {folder}")
        continue

    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for fname in tqdm(image_files, desc=f"Extracting features from {cls_name}"):
        path = os.path.join(folder, fname)

        # Read image
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Failed to load {path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Prepare input
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Extract features using ViT backbone only
        with torch.no_grad():
            vit_outputs = model.vit(**inputs)
            cls_token = vit_outputs.last_hidden_state[:, 0, :]   # [batch, 768]
            feat = cls_token.cpu().numpy().flatten()             # 768-dim vector

        # Store
        paths_list.append(path)
        labels_list.append(cls_idx)
        features_list.append(feat)

        # Save to SQLite (features as bytes)
        cursor.execute(
            "INSERT OR REPLACE INTO nodules (path, label, features) VALUES (?, ?, ?)",
            (path, cls_idx, feat.tobytes())
        )

# Commit DB changes
conn.commit()
conn.close()

# Add all features to FAISS index
if features_list:
    features_array = np.array(features_list).astype('float32')
    index.add(features_array)
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved with {index.ntotal} vectors")
else:
    print("No valid features extracted — index not created")

print(f"\nISR setup complete!")
print(f"Total entries processed: {len(paths_list)}")
print(f"Database: {DB_PATH}")
print(f"FAISS index: {INDEX_PATH}")