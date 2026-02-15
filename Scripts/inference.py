import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import cv2
import argparse
import numpy as np

MODEL_PATH = "Model/vit_fold4_best.pth"  # ← your best model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=3,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

class_names = ["Normal", "Benign", "Malignant"]

def predict(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Error loading image"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs).logits
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred = np.argmax(probs)
        conf = probs[pred]
    return class_names[pred], conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image")
    args = parser.parse_args()
    label, conf = predict(args.image)
    print(f"Prediction: {label} (Confidence: {conf:.4f})")