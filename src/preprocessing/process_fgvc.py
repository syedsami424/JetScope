import os
import shutil 
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from keras.applications.efficientnet import preprocess_input

RAW_DATA_DIR = Path("data/raw/data")
IMAGE_DIR = RAW_DATA_DIR / "images"
SPLIT_DIR = Path("data/processed")  # output directory
IMAGE_SIZE = (300, 300)

SPLITS = {
    "images_variant_train.txt": "train",
    "images_variant_val.txt": "val",
    "images_variant_test.txt": "test"
}

def load_variant_labels():
    variant_map = {}
    for fname in SPLITS.keys():
        with open(RAW_DATA_DIR / fname, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                image_id = parts[0]
                label = " ".join(parts[1:]).replace("/", "-")
                variant_map[image_id] = label
    return variant_map

def process_images(variant_map):
    for split_file, split_name in SPLITS.items():
        split_path = SPLIT_DIR / split_name
        if split_path.exists():
            shutil.rmtree(split_path)
        os.makedirs(split_path, exist_ok=True)
        
        with open(RAW_DATA_DIR / split_file, "r") as f:
            for line in tqdm(f, desc=f"Processing {split_name}"):
                parts = line.strip().split(" ")
                image_id = parts[0]
                label = " ".join(parts[1:]).replace("/", "-")
                src_img = IMAGE_DIR / f"{image_id}.jpg"
                dst_dir = split_path / label
                os.makedirs(dst_dir, exist_ok=True)
                dst_img = dst_dir / f"{image_id}.jpg"

                try:
                    img = Image.open(src_img).convert("RGB").resize(IMAGE_SIZE)
                    img_array = np.array(img).astype(np.float32)
                    img_array = preprocess_input(img_array)  # EfficientNet normalization
                    img_array = np.clip(((img_array - img_array.min()) / (img_array.max() - img_array.min())) * 255, 0, 255).astype(np.uint8)
                    Image.fromarray(img_array).save(dst_img)
                except Exception as e:
                    print(f"Failed to process {src_img}: {e}")

if __name__ == "__main__":
    variant_map = load_variant_labels()
    process_images(variant_map)
    print("Normalized images saved to data/processed/")
