# similarity_search.py
# Query the ISR system: predict class + find similar cases

import os
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import faiss
import sqlite3
import numpy as np
import cv2
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# ===================== CONFIGURATION =====================
MODEL_PATH = "Model/vit_fold8_best.pth"     # your best fold
DB_PATH = "Database/lung_nodule.db"
INDEX_PATH = "Database/faiss_index.index"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 5                                    # number of similar cases to show

class_names = ["Normal", "Benign", "Malignant"]

# ==========================================================

# Load ViT model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=3,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Load FAISS index
index = faiss.read_index(INDEX_PATH)
print(f"FAISS index loaded with {index.ntotal} vectors")

# Connect to DB
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

def predict_and_search(image_path, k=TOP_K):
    # Read & preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Model prediction
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]

        # Get feature (CLS token)
        vit_outputs = model.vit(**inputs)
        feat = vit_outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [1, 768]

    print(f"\nPrediction: {class_names[pred_class]} (Confidence: {confidence:.4f})")

    # Search similar
    distances, indices = index.search(feat.astype('float32'), k)

    print(f"\nTop {k} most similar cases:")
    similar_info = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:
            continue  # invalid index

        cursor.execute("SELECT path, label FROM nodules WHERE rowid = ?", (idx + 1,))
        result = cursor.fetchone()
        if result:
            sim_path, sim_label = result
            similar_info.append((sim_path, class_names[sim_label], dist))
            print(f"  {rank+1}. Distance: {dist:.4f} | Label: {class_names[sim_label]}")
            print(f"     Path: {sim_path}")

    conn.close()
    return similar_info

def show_similar_images(similar_info, query_img_path):
    n = len(similar_info) + 1
    fig, ax_array = plt.subplots(1, n, figsize=(3*n, 5))
    
    # Make ax_array always a list/array
    axes = ax_array if n > 1 else [ax_array]  # ← important fix

    # Query
    query_img = cv2.cvtColor(cv2.imread(query_img_path), cv2.COLOR_BGR2RGB)
    axes[0].imshow(query_img)
    axes[0].set_title("Query Image")
    axes[0].axis('off')

    # Similar
    for i, (path, label, dist) in enumerate(similar_info, 1):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].set_title(f"Rank {i}\n{label}\nDist: {dist:.4f}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query similar lung nodules")
    parser.add_argument("image", type=str, help="Path to query image (PNG/JPG)")
    parser.add_argument("--k", type=int, default=5, help="Number of similar cases")
    args = parser.parse_args()

    similar = predict_and_search(args.image, args.k)
    show_similar_images(similar, args.image)