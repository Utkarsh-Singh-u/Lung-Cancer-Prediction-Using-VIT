import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
import cv2
import shutil
from tqdm import tqdm

# ====================== CONFIGURATION ======================
RAW_DATA_DIR = "Data/raw/subset9"           # ← change for each new subset
ANNOTATIONS_FILE = "Data/raw/annotations.csv"
CANDIDATES_FILE = "Data/raw/candidates_V2.csv"
OUTPUT_DIR = "Data/processed/patches"

PATCH_SIZE_MM = 50
TARGET_SIZE = 224

HU_MIN = -1000
HU_MAX = 400

NUM_NEGATIVES_PER_SCAN = 3
MIN_DISTANCE_TO_NODULE_MM = 25.0
DIAM_BENIGN_THRESHOLD = 10.0
# ===========================================================

# def clean_output_dir():
#     if os.path.exists(OUTPUT_DIR):
#         print(f"Cleaning old data in {OUTPUT_DIR}...")
#         shutil.rmtree(OUTPUT_DIR)
#     os.makedirs(os.path.join(OUTPUT_DIR, "normal"), exist_ok=True)
#     os.makedirs(os.path.join(OUTPUT_DIR, "benign"), exist_ok=True)
#     os.makedirs(os.path.join(OUTPUT_DIR, "malignant"), exist_ok=True)

def normalize_hu(image_numpy):
    image_numpy = np.clip(image_numpy, HU_MIN, HU_MAX)
    image_numpy = (image_numpy - HU_MIN) / (HU_MAX - HU_MIN + 1e-8)
    return (image_numpy * 255).astype(np.uint8)

def extract_patch(image_array, center_point_idx, spacing):
    c_x, c_y, c_z = center_point_idx
    z_idx = int(np.round(c_z))
    y_idx = int(np.round(c_y))
    x_idx = int(np.round(c_x))

    num_z, height, width = image_array.shape
    z_idx = np.clip(z_idx, 0, num_z - 1)

    # Use average spacing for square physical patch
    avg_spacing = (spacing[0] + spacing[1]) / 2
    patch_rad_vox = int((PATCH_SIZE_MM / 2) / avg_spacing)
    patch_rad_vox = max(16, patch_rad_vox)

    y_min = max(0, y_idx - patch_rad_vox)
    y_max = min(height, y_idx + patch_rad_vox + 1)
    x_min = max(0, x_idx - patch_rad_vox)
    x_max = min(width, x_idx + patch_rad_vox + 1)

    patch = image_array[z_idx, y_min:y_max, x_min:x_max]

    desired_size = 2 * patch_rad_vox
    curr_h, curr_w = patch.shape

    if curr_h < desired_size or curr_w < desired_size:
        pad_top    = max(0, patch_rad_vox - (y_idx - y_min))
        pad_bottom = max(0, patch_rad_vox - (y_max - y_idx - 1))
        pad_left   = max(0, patch_rad_vox - (x_idx - x_min))
        pad_right  = max(0, patch_rad_vox - (x_max - x_idx - 1))

        patch = np.pad(patch,
                       ((pad_top, pad_bottom), (pad_left, pad_right)),
                       mode='constant', constant_values=HU_MIN)

    return patch, z_idx

def process_subset():
    # clean_output_dir()

    df_ann = pd.read_csv(ANNOTATIONS_FILE)
    df_cand = pd.read_csv(CANDIDATES_FILE)

    mhd_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".mhd")]
    print(f"Found {len(mhd_files)} scans in {RAW_DATA_DIR}")

    total_saved = {"normal": 0, "benign": 0, "malignant": 0}
    skipped = 0

    for mhd_file in tqdm(mhd_files):
        series_uid = mhd_file.replace(".mhd", "")
        mhd_path = os.path.join(RAW_DATA_DIR, mhd_file)

        try:
            itk_image = sitk.ReadImage(mhd_path)
            image_array = sitk.GetArrayFromImage(itk_image)
            spacing = np.array(itk_image.GetSpacing())
        except Exception as e:
            print(f"Error reading {mhd_file}: {e}")
            continue

        # Positive samples
        nodules = df_ann[df_ann["seriesuid"] == series_uid]
        for i, nodule in nodules.iterrows():
            point_mm = (nodule["coordX"], nodule["coordY"], nodule["coordZ"])
            point_idx = itk_image.TransformPhysicalPointToContinuousIndex(point_mm)

            patch, used_z = extract_patch(image_array, point_idx, spacing)
            if patch is None or patch.size == 0:
                skipped += 1
                continue

            patch_norm = normalize_hu(patch)
            patch_resized = cv2.resize(patch_norm, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_CUBIC)

            label = "benign" if nodule["diameter_mm"] < DIAM_BENIGN_THRESHOLD else "malignant"
            save_path = os.path.join(OUTPUT_DIR, label, f"{series_uid}_z{used_z:03d}_n{i}.png")
            cv2.imwrite(save_path, patch_resized)
            total_saved[label] += 1

        # Negative samples
        negatives = df_cand[(df_cand["seriesuid"] == series_uid) & (df_cand["class"] == 0)]
        nodule_coords = nodules[["coordX", "coordY", "coordZ"]].values if len(nodules) > 0 else np.empty((0,3))

        valid_neg = []
        for _, row in negatives.iterrows():
            cand_coord = np.array([row["coordX"], row["coordY"], row["coordZ"]])
            if len(nodule_coords) > 0:
                dists = np.linalg.norm(nodule_coords - cand_coord, axis=1)
                if np.min(dists) < MIN_DISTANCE_TO_NODULE_MM:
                    continue
            valid_neg.append(row)
            if len(valid_neg) >= NUM_NEGATIVES_PER_SCAN:
                break

        for i, neg in enumerate(valid_neg):
            point_mm = (neg["coordX"], neg["coordY"], neg["coordZ"])
            point_idx = itk_image.TransformPhysicalPointToContinuousIndex(point_mm)

            patch, used_z = extract_patch(image_array, point_idx, spacing)
            if patch is None or patch.size == 0:
                skipped += 1
                continue

            patch_norm = normalize_hu(patch)
            patch_resized = cv2.resize(patch_norm, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_CUBIC)

            save_path = os.path.join(OUTPUT_DIR, "normal", f"{series_uid}_z{used_z:03d}_neg{i}.png")
            cv2.imwrite(save_path, patch_resized)
            total_saved["normal"] += 1

    print("\nProcessing Complete!")
    print("Patches saved:")
    for k, v in total_saved.items():
        print(f"  {k}: {v}")
    print(f"Skipped invalid patches: {skipped}")
    print(f"Check your data at: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_subset()