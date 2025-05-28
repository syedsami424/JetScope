import os
import shutil 
from PIL import Image
from pathlib import Path
from tqdm import tqdm

RAW_DATA_DIR = Path("data/raw/data")
IMAGE_DIR = RAW_DATA_DIR / "images"
SPLIT_DIR = Path("data/processed") #output directory

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
                # print(image_id)
                label = " ".join(parts[1:]).replace("/", "-")
                variant_map[image_id] = label
    return variant_map

def process_images(variant_map):
    for split_file, split_name in SPLITS.items():
        split_path = SPLIT_DIR / split_name
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

                ## RESIZE IMAGES
                try:
                    with Image.open(src_img) as img:
                        img = img.convert("RGB")
                        img = img.resize(IMAGE_SIZE)
                        img.save(dst_img)
                except Exception as e:
                    print(f"Failed to process {src_img}: {e}")


if __name__ == "__main__":
    variant_map = load_variant_labels()
    process_images(variant_map)
    print("Images saved to data/processed/")