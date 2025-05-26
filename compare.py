import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio

# Paths
png_dir = Path("./volume-data/chameleon/images")
jpg_dir = Path("./renders")
comparison_dir = Path("./comparison")

# Delete existing comparison directory if it exists and recreate it
if comparison_dir.exists():
    shutil.rmtree(comparison_dir)
os.makedirs(comparison_dir, exist_ok=True)

# Find all PNG files
png_files = list(png_dir.glob("*.png"))

# Dictionary to store PSNR values
psnr_values = {}

# Process each PNG file
for png_path in png_files:
    # Get the filename without extension
    base_name = png_path.stem

    # Find corresponding JPG file
    jpg_path = jpg_dir / f"{base_name}.jpg"

    # Check if corresponding JPG exists
    if jpg_path.exists():
        # Read images
        png_img = io.imread(png_path)
        jpg_img = io.imread(jpg_path)

        # Handle channel differences - convert PNG from RGBA to RGB if needed
        if png_img.shape[-1] == 4 and jpg_img.shape[-1] == 3:
            # Remove alpha channel from PNG
            png_img = png_img[..., :3]
            print(f"Removed alpha channel from {base_name}.png")

        # Calculate PSNR
        try:
            # Make sure images have same dimensions for PSNR calculation
            if png_img.shape == jpg_img.shape:
                psnr = peak_signal_noise_ratio(png_img, jpg_img)
                psnr_values[base_name] = psnr
            else:
                print(
                    f"Warning: {base_name} images still have different shapes after alpha removal. PNG: {png_img.shape}, JPG: {jpg_img.shape}"
                )
                psnr_values[base_name] = None
        except Exception as e:
            print(f"Error calculating PSNR for {base_name}: {e}")
            psnr_values[base_name] = None

        # Create side-by-side comparison
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(png_img)
        plt.title(f"PNG: {base_name}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(jpg_img)
        plt.title(f"JPG: {base_name}")
        plt.axis("off")

        plt.suptitle(
            f"PSNR: {psnr_values[base_name]:.2f}"
            if psnr_values[base_name] is not None
            else "PSNR: N/A"
        )
        plt.tight_layout()

        # Save comparison image
        comparison_path = comparison_dir / f"{base_name}_comparison.png"
        plt.savefig(comparison_path)
        plt.close()

        print(f"Processed: {base_name}")
    else:
        print(f"Warning: No corresponding JPG found for {png_path}")

# Save PSNR values to JSON file
with open(comparison_dir / "psnr_values.json", "w") as f:
    json.dump(psnr_values, f, indent=2)

print(f"Completed! Comparison images and PSNR values saved to {comparison_dir}")
