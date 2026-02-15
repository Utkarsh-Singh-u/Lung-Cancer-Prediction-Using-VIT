# mhd_to_png_converter.py
# Converts every slice of every .mhd file in a subset to PNG images
# No cropping, no patches — full slices only

import os
import SimpleITK as sitk
import numpy as np
import cv2
from tqdm import tqdm
import argparse

# ====================== CONFIGURATION ======================
# Change this to the subset you want to convert
SUBSET_FOLDER = "Data/raw/subset0"          # ← change to "subset1", "subset2", etc.

# Output folder (will create subfolder per scan)
OUTPUT_BASE = "Data/raw_slices_png"

# HU window for lung CT (standard)
HU_MIN = -1000
HU_MAX = 400

# ===========================================================

def normalize_to_png(image_slice):
    """ Clip HU range and convert to uint8 for PNG """
    image_slice = np.clip(image_slice, HU_MIN, HU_MAX)
    image_slice = (image_slice - HU_MIN) / (HU_MAX - HU_MIN + 1e-8)
    image_slice = (image_slice * 255).astype(np.uint8)
    return image_slice

def convert_mhd_to_png(mhd_path, output_dir):
    """ Convert one .mhd file to multiple PNG slices """
    try:
        itk_image = sitk.ReadImage(mhd_path)
        array = sitk.GetArrayFromImage(itk_image)  # shape: (z, y, x)
        series_uid = os.path.basename(mhd_path).replace(".mhd", "")
        
        scan_dir = os.path.join(output_dir, series_uid)
        os.makedirs(scan_dir, exist_ok=True)
        
        saved_count = 0
        for z in range(array.shape[0]):
            slice_img = array[z, :, :]
            png_img = normalize_to_png(slice_img)
            
            # Optional: flip vertically if orientation looks wrong (common in some CTs)
            # png_img = cv2.flip(png_img, 0)
            
            filename = f"slice_z{z:04d}.png"
            save_path = os.path.join(scan_dir, filename)
            cv2.imwrite(save_path, png_img)
            saved_count += 1
        
        return saved_count
    
    except Exception as e:
        print(f"Error processing {mhd_path}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Convert .mhd CT volumes to PNG slices")
    parser.add_argument("--subset", type=str, default=SUBSET_FOLDER,
                        help="Path to the subset folder containing .mhd files")
    args = parser.parse_args()

    subset_dir = args.subset
    if not os.path.isdir(subset_dir):
        print(f"Error: Folder not found: {subset_dir}")
        return

    mhd_files = [f for f in os.listdir(subset_dir) if f.lower().endswith(".mhd")]
    if not mhd_files:
        print(f"No .mhd files found in {subset_dir}")
        return

    print(f"Found {len(mhd_files)} .mhd files in {subset_dir}")
    print(f"Output will be saved in: {OUTPUT_BASE}\n")

    total_slices = 0

    for mhd_file in tqdm(mhd_files, desc="Converting scans"):
        mhd_path = os.path.join(subset_dir, mhd_file)
        slices_saved = convert_mhd_to_png(mhd_path, OUTPUT_BASE)
        total_slices += slices_saved
        print(f"  Saved {slices_saved} slices from {mhd_file}")

    print(f"\nConversion finished!")
    print(f"Total slices converted: {total_slices}")
    print(f"All PNG files saved under: {OUTPUT_BASE}")

if __name__ == "__main__":
    main()