import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import config
from src.data.preprocess import extract_blockwise_fft, extract_gradcam_blockwise_fft
from src.models.fftnet import FFTNet
from src.models.xgboost_model import build_xgboost_model
from src.models.resnet import build_resnet18
from src.training.trainer import train_model
from src.utils.cam import GradCAM, load_or_compute_cam
from src.utils.metrics import evaluate_pytorch_model, print_evaluation_metrics

def load_split_blockwise(split_dir, split_name, class_names, device, use_pca=False, use_gradcam=False, gradcam_instance=None):
    X, y = [], []
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(split_dir, class_name)
        if not os.path.exists(class_path):
            continue
            
        image_files = os.listdir(class_path)
        print(f"  - Loading {split_name} Class {class_name} ({len(image_files)} images)...", flush=True)
        
        for img_name in image_files:
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is None: continue

            if use_gradcam and gradcam_instance:
                cam = load_or_compute_cam(
                    image, label, img_path, split_name, class_name, 
                    "cam_cache_gradcam_blockwise", gradcam_instance, config.CAM_TARGET_MODE
                )
                feat = extract_gradcam_blockwise_fft(
                    image, cam, config.IMG_SIZE, config.BLOCK_SIZE, 
                    raw=use_pca, min_attention=config.MIN_ATTENTION, attention_strength=config.ATTENTION_STRENGTH
                )
            else:
                feat = extract_blockwise_fft(image, config.IMG_SIZE, config.BLOCK_SIZE, raw=use_pca)
                
            X.append(feat)
            y.append(label)
    return np.array(X), np.array(y)

def main():
    parser = argparse.ArgumentParser(description="Train Blockwise FFT models")
    parser.add_argument("--use-pca", action="store_true", help="Use raw magnitudes and PCA")
    parser.add_argument("--use-gradcam", action="store_true", help="Use Grad-CAM attention")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | PCA: {args.use_pca} | Grad-CAM: {args.use_gradcam}")

    prefix = "blockwise"
    if args.use_gradcam: prefix = f"gradcam_{prefix}"
    if args.use_pca: prefix = f"{prefix}_pca"
    
    # Check if this script's work is already done
    model_path = os.path.join(config.PYTORCH_MODEL_DIR, f"fft_nn_{prefix}.pth")
    if os.path.exists(model_path):
        print(f"Blockwise FFT ({prefix}) model already found at {model_path}. Skipping step...")
        return


    train_dir = os.path.join(config.DATASET_DIR, "train")
    val_dir = os.path.join(config.DATASET_DIR, "val")
    test_dir = os.path.join(config.DATASET_DIR, "test")

    class_names = sorted(os.listdir(train_dir))
    num_classes = len(class_names)
    
    gradcam_instance = None
    if args.use_gradcam:
        print("Loading Pretrained ResNet for Grad-CAM...")
        resnet = build_resnet18(num_classes).to(device)
        resnet_ckpt = os.path.join(config.PYTORCH_MODEL_DIR, "best_resnet18.pth")
        
        try:
            checkpoint = torch.load(resnet_ckpt, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(resnet_ckpt, map_location=device)
            
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        resnet.load_state_dict(checkpoint)
        resnet.eval()
        
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(device)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).to(device)
        gradcam_instance = GradCAM(resnet, "layer4", config.IMG_SIZE, device, imagenet_mean, imagenet_std)

    print("Extracting features...")
    X_train, y_train = load_split_blockwise(train_dir, "train", class_names, device, args.use_pca, args.use_gradcam, gradcam_instance)
    X_val, y_val = load_split_blockwise(val_dir, "val", class_names, device, args.use_pca, args.use_gradcam, gradcam_instance)
    X_test, y_test = load_split_blockwise(test_dir, "test", class_names, device, args.use_pca, args.use_gradcam, gradcam_instance)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    prefix = "blockwise"
    if args.use_gradcam: prefix = f"gradcam_{prefix}"
    if args.use_pca: prefix = f"{prefix}_pca"
    
    joblib.dump(scaler, os.path.join(config.PREPROCESSOR_DIR, f"{prefix}_scaler.pkl"))

    if args.use_pca:
        print("Running PCA...")
        pca = PCA(n_components=min(config.PCA_COMPONENTS, X_train.shape[0], X_train.shape[1]), random_state=config.SEED)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)
        joblib.dump(pca, os.path.join(config.PREPROCESSOR_DIR, f"{prefix}_pca.pkl"))

    # XGBoost
    print("\nTraining XGBoost...")
    xgb_model = build_xgboost_model(num_classes, config.XGB_PARAMS, str(device))
    try:
        xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], early_stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=True)
    except TypeError:
        xgb_model.set_params(early_stopping_rounds=config.EARLY_STOPPING_ROUNDS)
        xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True)

    y_pred = xgb_model.predict(X_test)
    print("XGBoost Test Accuracy:", accuracy_score(y_test, y_pred))
    
    results = xgb_model.evals_result()
    if results and "validation_1" in results:
        pd.DataFrame(results["validation_1"]).to_csv(os.path.join(config.LOG_DIR, f"xgb_{prefix}_log.csv"), index=False)
    xgb_model.save_model(os.path.join(config.XGBOOST_MODEL_DIR, f"xgboost_{prefix}.json"))

    # NN Model
    print("\nTraining FFTNet...")
    model = FFTNet(X_train.shape[1], num_classes).to(device)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())

    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.FFT_BLOCKWISE_WD)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    model, history, best_val_acc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=config.EPOCHS,
        patience=config.PATIENCE,
        device=device
    )

    pd.DataFrame(history).to_csv(os.path.join(config.LOG_DIR, f"fft_nn_{prefix}_log.csv"), index=False)
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": X_train.shape[1],
        "num_classes": num_classes,
        "class_names": class_names,
        "best_val_acc": best_val_acc
    }, os.path.join(config.PYTORCH_MODEL_DIR, f"fft_nn_{prefix}.pth"))

    print("Evaluating NN on Test Set...")
    _, _, _, preds, all_labels = evaluate_pytorch_model(model, test_loader, device)
    print_evaluation_metrics(all_labels, preds, target_names=class_names)

if __name__ == "__main__":
    main()
