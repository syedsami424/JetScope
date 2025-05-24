import os
import shutil 
from PIL import Image
from pathlib import Path
from tqdm import tqdm

RAW_DATA_DIR = Path("data/raw/data")
IMAGE_DIR = RAW_DATA_DIR / "images"
SPLIT_DIR = Path("data/processed") #output directory

IMAGE_SIZE = (224, 244)

SPLITS = {
    "images_variant_train.txt": "train",
    "images_variant_val.txt": "val",
    "images_variant_test.txt": "test"
}

