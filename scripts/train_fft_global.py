import os
import sys
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
from src.data.preprocess import extract_global_fft
from src.models.fftnet import FFTGlobalNet
from src.models.xgboost_model import build_xgboost_model
from src.training.trainer import train_model
from src.utils.metrics import evaluate_pytorch_model, print_evaluation_metrics

def load_split_global(split_dir, class_names, device):
    X, y = [], []
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(split_dir, class_name)
        if not os.path.exists(class_path):
            continue
        
        img_list = os.listdir(class_path)
        print(f"  - Loading {os.path.basename(split_dir)} Class {class_name} ({len(img_list)} images)...", flush=True)
        
        for img_name in img_list:
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is None: continue
            feat = extract_global_fft(image, img_size=config.IMG_SIZE, device=device)
            X.append(feat)
            y.append(label)
    return np.array(X), np.array(y)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if this script's work is already done
    model_path = os.path.join(config.PYTORCH_MODEL_DIR, "fft_global_nn.pth")
    xgb_model_path = os.path.join(config.XGBOOST_MODEL_DIR, "xgboost_global.json")
    if os.path.exists(model_path) and os.path.exists(xgb_model_path):
        print(f"Global FFT models already found. Skipping step...")
        return

    train_dir = os.path.join(config.VGGFACE_DATASET_DIR, "train")
    if not os.path.exists(train_dir):
        print(f"Warning: Dataset not found at {train_dir}. Trying STL-10 dataset dir.")
        train_dir = os.path.join(config.DATASET_DIR, "train")
        val_dir = os.path.join(config.DATASET_DIR, "val")
        test_dir = os.path.join(config.DATASET_DIR, "test")
    else:
        val_dir = os.path.join(config.VGGFACE_DATASET_DIR, "val")
        test_dir = os.path.join(config.VGGFACE_DATASET_DIR, "test")
        
    class_names = sorted(os.listdir(train_dir))
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")

    print("Extracting features...")
    X_train, y_train = load_split_global(train_dir, class_names, device)
    X_val, y_val = load_split_global(val_dir, class_names, device)
    X_test, y_test = load_split_global(test_dir, class_names, device)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(config.PREPROCESSOR_DIR, "global_fft_scaler.pkl"))

    print("Running PCA...")
    pca = PCA(n_components=min(config.PCA_COMPONENTS, X_train.shape[0], X_train.shape[1]), random_state=config.SEED)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    joblib.dump(pca, os.path.join(config.PREPROCESSOR_DIR, "global_fft_pca.pkl"))

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
        pd.DataFrame(results["validation_1"]).to_csv(os.path.join(config.LOG_DIR, "xgb_global_log.csv"), index=False)
    xgb_model.save_model(os.path.join(config.XGBOOST_MODEL_DIR, "xgboost_global.json"))

    # NN Model
    print("\nTraining FFTGlobalNet...")
    model = FFTGlobalNet(X_train.shape[1], num_classes).to(device)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())

    optimizer = optim.Adam(model.parameters(), lr=config.FFT_GLOBAL_LR, weight_decay=config.FFT_GLOBAL_WD)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    model, history, best_val_acc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=None,
        epochs=config.EPOCHS,
        patience=config.PATIENCE,
        device=device
    )

    pd.DataFrame(history).to_csv(os.path.join(config.LOG_DIR, "fft_global_nn_log.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(config.PYTORCH_MODEL_DIR, "fft_global_nn.pth"))

    print("Evaluating NN on Test Set...")
    _, _, _, preds, all_labels = evaluate_pytorch_model(model, test_loader, device)
    print_evaluation_metrics(all_labels, preds, target_names=class_names)

if __name__ == "__main__":
    main()
