import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
import numpy as np
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score
from torch.cuda.amp import autocast, GradScaler
import pandas as pd

# ===================== CONFIGURATION =====================
DATA_ROOT = "Data/processed/patches"          # your local folder
MODEL_SAVE_DIR = "Model"                      # where to save .pth files
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32               # safe for RTX 3050 (4–6 GB VRAM)
MAX_EPOCHS = 25
LEARNING_RATE = 1e-4
PATIENCE = 8                  # early stopping
NUM_WORKERS = 4               # adjust down to 2 if CPU bottleneck
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Strong augmentation for training
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.GaussianBlur(p=0.15),
    A.Normalize(mean=0.5, std=0.5),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=0.5, std=0.5),
    ToTensorV2()
])

# ==============================================
# Dataset Class
# ==============================================

class LungDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            aug = self.transform(image=img)
            img = aug['image']

        return img, self.labels[idx]

# ==============================================
# Load all data + assign fold (hash-based)
# ==============================================

def load_all_data():
    classes = ["normal", "benign", "malignant"]
    class_to_idx = {"normal": 0, "benign": 1, "malignant": 2}

    all_paths = []
    all_labels = []
    all_series = []

    for cls in classes:
        folder = os.path.join(DATA_ROOT, cls)
        for fname in os.listdir(folder):
            if fname.lower().endswith(".png"):
                path = os.path.join(folder, fname)
                series_uid = fname.split('_')[0]  # extract series_uid from filename
                all_paths.append(path)
                all_labels.append(class_to_idx[cls])
                all_series.append(series_uid)

    df = pd.DataFrame({
        'path': all_paths,
        'label': all_labels,
        'series_uid': all_series
    })

    # Assign fold deterministically (0–9)
    df['fold'] = df['series_uid'].apply(lambda x: hash(x) % 10)

    print("Dataset loaded:")
    print(df['label'].value_counts())
    print("\nFold distribution:\n", df['fold'].value_counts())

    return df

# ==============================================
# Train one fold
# ==============================================
train_loss=0

def train_one_fold(fold_id, train_df, val_df):
    train_ds = LungDataset(train_df['path'].values, train_df['label'].values, train_transform)
    val_ds   = LungDataset(val_df['path'].values,   val_df['label'].values,   val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    model.to(DEVICE)

    # Class weights for imbalance
    class_counts = np.bincount(train_df['label'])
    weights = 1.0 / class_counts
    weights = torch.tensor(weights / weights.sum(), dtype=torch.float).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scaler = GradScaler()

    best_val_acc = 0
    patience_counter = 0
    best_epoch = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0
        for imgs, lbls in tqdm(train_loader, desc=f"Fold {fold_id} Epoch {epoch+1}/{MAX_EPOCHS}"):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs).logits
                loss = criterion(outputs, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_true = [], []
        val_loss = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                with autocast():
                    outputs = model(imgs).logits
                    loss = criterion(outputs, lbls)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(lbls.cpu().numpy())

        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average='weighted')

        print(f"Fold {fold_id} | Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} "
              f"| Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # Save best model for this fold
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"vit_fold{fold_id}_best.pth"))
            print(f"  → Saved best model for fold {fold_id} (epoch {best_epoch}, acc {val_acc:.4f})")

        # Early stopping
        if val_acc < best_val_acc:  # no improvement
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} for fold {fold_id}")
                break
        else:
            patience_counter = 0

    return best_val_acc

# ==============================================
# Main 10-fold loop
# ==============================================

def main():
    df = load_all_data()

    fold_accuracies = []

    for fold in range(10):
        print(f"\n===== Fold {fold} =====")
        train_df = df[df['fold'] != fold]
        val_df   = df[df['fold'] == fold]

        acc = train_one_fold(fold, train_df, val_df)
        fold_accuracies.append(acc)

    print("\n10-fold cross-validation finished!")
    print("Fold accuracies:", fold_accuracies)
    print(f"Mean validation accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Standard deviation: {np.std(fold_accuracies):.4f}")

if __name__ == "__main__":
    main()