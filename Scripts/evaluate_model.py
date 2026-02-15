import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import cv2
import os
from tqdm import tqdm

# Config
MODEL_PATH = "Model/vit_fold4_best.pth"  # ← change to your best fold
DATA_ROOT = "Data/processed/patches"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

class_names = ["Normal", "Benign", "Malignant"]

# Load model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=3,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Collect all test paths/labels (use all data or make a small test set)
all_paths, all_labels = [], []
for cls_idx, cls in enumerate(["normal", "benign", "malignant"]):
    folder = os.path.join(DATA_ROOT, cls)
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]
    all_paths.extend(paths)
    all_labels.extend([cls_idx] * len(paths))

print(f"Evaluating on {len(all_paths)} images")

# Inference
preds = []
true_labels = []

with torch.no_grad():
    for path, lbl in tqdm(zip(all_paths, all_labels), total=len(all_paths)):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = model(**inputs).logits
        pred = torch.argmax(outputs, dim=1).item()
        preds.append(pred)
        true_labels.append(lbl)

# Metrics
acc = accuracy_score(true_labels, preds)
print(f"Overall Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(true_labels, preds, target_names=class_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(true_labels, preds)
print(cm)