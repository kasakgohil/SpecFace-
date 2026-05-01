import os
import sys
import numpy as np
import cv2
from PIL import Image
import shutil
import random
from tqdm import tqdm
import errno

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import config
from src.data.preprocess import apply_histeq, apply_denoise

HEIGHT, WIDTH, DEPTH = config.IMG_SIZE, config.IMG_SIZE, 3
SIZE = HEIGHT * WIDTH * DEPTH

def read_labels(path_to_labels):
    with open(path_to_labels, 'rb') as f:
        return np.fromfile(f, dtype=np.uint8)

def read_all_images(path_to_data):
    with open(path_to_data, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, config.IMG_SIZE, config.IMG_SIZE))
        images = np.transpose(images, (0, 3, 2, 1))
        return images

def save_images(images, labels, base_dir, split_type):
    for i in tqdm(range(len(images)), desc=f"Extracting {split_type}"):
        image = images[i]
        label = labels[i]
        directory = os.path.join(base_dir, split_type, str(label))
        os.makedirs(directory, exist_ok=True)
        
        filename = os.path.join(directory, f"{i}.png")
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def create_split_and_preprocess(raw_dir, preprocessed_dir):
    train_dir = os.path.join(raw_dir, "train")
    test_dir = os.path.join(raw_dir, "test")

    train_out = os.path.join(preprocessed_dir, "train")
    val_out = os.path.join(preprocessed_dir, "val")
    test_out = os.path.join(preprocessed_dir, "test")

    classes = [str(i) for i in range(1, 11)]

    for split in [train_out, val_out, test_out]:
        for cls in classes:
            os.makedirs(os.path.join(split, cls), exist_ok=True)

    train_ratio = config.TRAIN_RATIO
    val_ratio = config.VAL_RATIO
    random.seed(config.SEED)

    for cls in classes:
        all_images = []
        
        # Paths to extract
        if os.path.exists(os.path.join(train_dir, cls)):
            for img in os.listdir(os.path.join(train_dir, cls)):
                all_images.append(os.path.join(train_dir, cls, img))

        if os.path.exists(os.path.join(test_dir, cls)):
            for img in os.listdir(os.path.join(test_dir, cls)):
                all_images.append(os.path.join(test_dir, cls, img))

        random.shuffle(all_images)
        total = len(all_images)

        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)

        train_imgs = all_images[:train_count]
        val_imgs = all_images[train_count:train_count + val_count]
        test_imgs = all_images[train_count + val_count:]

        def process_and_save(img_paths, out_dir):
            for i, img_path in enumerate(img_paths):
                img = cv2.imread(img_path)
                if img is None: continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
                gray_eq = apply_histeq([gray])[0]
                gray_denoised = apply_denoise([gray_eq], ksize=config.GAUSSIAN_KSIZE)[0]
                
                out_path = os.path.join(out_dir, cls, f"{i}.png")
                img_uint8 = (gray_denoised * 255).astype(np.uint8)
                cv2.imwrite(out_path, img_uint8)

        print(f"Processing Class {cls}...")
        process_and_save(train_imgs, train_out)
        process_and_save(val_imgs, val_out)
        process_and_save(test_imgs, test_out)

if __name__ == "__main__":
    raw_data_dir = config.DATA_DIR
    train_x_path = os.path.join(raw_data_dir, "train_X.bin")
    train_y_path = os.path.join(raw_data_dir, "train_y.bin")
    test_x_path = os.path.join(raw_data_dir, "test_X.bin")
    test_y_path = os.path.join(raw_data_dir, "test_y.bin")
    
    # Check if data is already processed
    if os.path.exists(os.path.join(config.DATASET_DIR, "train", "1")):
        print(f"Preprocessed data already found in {config.DATASET_DIR}. Skipping data setup...")
        sys.exit(0)

        print("Extracting training binaries...")
        train_labels = read_labels(train_y_path)
        train_images = read_all_images(train_x_path)
        save_images(train_images, train_labels, raw_data_dir, "train")
        
    if os.path.exists(test_x_path):
        print("Extracting testing binaries...")
        test_labels = read_labels(test_y_path)
        test_images = read_all_images(test_x_path)
        save_images(test_images, test_labels, raw_data_dir, "test")
        
    print("Preprocessing and splitting data...")
    create_split_and_preprocess(raw_data_dir, config.DATASET_DIR)
    print("Data Setup Complete!")
