import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, cohen_kappa_score, log_loss, 
    roc_curve, auc, confusion_matrix
)
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.data.dataset import get_torchvision_datasets
from src.data.transforms import get_resnet_val_transforms
from src.data.preprocess import extract_global_fft, extract_blockwise_fft
from src.models.resnet import build_resnet18
from src.models.fftnet import FFTNet, FFTGlobalNet
from src.models.xgboost_model import build_xgboost_model

def load_test_images():
    print("Loading test images...")
    transform = get_resnet_val_transforms()
    # Note: data/raw/final_split seems to be where the actual split images are
    data_dir = os.path.join("data", "raw", "final_split")
    if not os.path.exists(data_dir):
        data_dir = config.DATASET_DIR # Fallback
        
    _, _, test_dataset = get_torchvision_datasets(data_dir, transform)
    return test_dataset

def get_predictions_pytorch(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Inference"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def get_predictions_xgb(model, X_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    return preds, probs

def calculate_metrics(y_true, y_pred, y_prob):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['log_loss'] = log_loss(y_true, y_prob)
    return metrics

def plot_roc_curves(all_results, class_names, save_path):
    plt.figure(figsize=(10, 8))
    for model_name, data in all_results.items():
        y_true = data['y_true']
        y_prob = data['y_prob']
        
        # For multi-class ROC, we typically do One-vs-Rest or micro-average
        # Here we do micro-average for simplicity in comparison
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Micro-averaged)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(save_path)
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_dataset = load_test_images()
    class_names = test_dataset.classes
    num_classes = len(class_names)
    
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Extract features once for FFT models to save time
    print("Extracting FFT features for comparison...")
    X_global = []
    X_blockwise = []
    y_test = []
    
    # We'll use a small subset or the full test set depending on size
    # For now, let's assume we use the full test set
    for img_path, label in tqdm(test_dataset.imgs, desc="Feature Extraction"):
        image = cv2.imread(img_path)
        if image is None: continue
        
        feat_global = extract_global_fft(image, config.IMG_SIZE, device)
        feat_blockwise = extract_blockwise_fft(image, config.IMG_SIZE, config.BLOCK_SIZE, device)
        
        X_global.append(feat_global)
        X_blockwise.append(feat_blockwise)
        y_test.append(label)
        
    X_global = np.array(X_global)
    X_blockwise = np.array(X_blockwise)
    y_test = np.array(y_test)
    
    all_results = {}
    
    # 1. ResNet-18
    print("\n--- Evaluating ResNet-18 ---")
    model_resnet = build_resnet18(num_classes).to(device)
    resnet_path = os.path.join(config.PYTORCH_MODEL_DIR, "best_resnet18.pth")
    if os.path.exists(resnet_path):
        checkpoint = torch.load(resnet_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        model_resnet.load_state_dict(checkpoint)
        y_true, y_pred, y_prob = get_predictions_pytorch(model_resnet, test_loader, device)
        all_results['ResNet-18'] = {
            'metrics': calculate_metrics(y_true, y_pred, y_prob),
            'y_true': y_true,
            'y_prob': y_prob,
            'y_pred': y_pred
        }
    else:
        print(f"Skipping ResNet-18 (not found at {resnet_path})")

    # 2. FFT Global NN
    print("\n--- Evaluating FFT Global NN ---")
    scaler_g = joblib.load(os.path.join(config.PREPROCESSOR_DIR, "global_fft_scaler.pkl"))
    pca_g = joblib.load(os.path.join(config.PREPROCESSOR_DIR, "global_fft_pca.pkl"))
    X_g_proc = pca_g.transform(scaler_g.transform(X_global))
    
    model_g_nn = FFTGlobalNet(X_g_proc.shape[1], num_classes).to(device)
    g_nn_path = os.path.join(config.PYTORCH_MODEL_DIR, "fft_global_nn.pth")
    if os.path.exists(g_nn_path):
        model_g_nn.load_state_dict(torch.load(g_nn_path, map_location=device))
        
        X_g_t = torch.tensor(X_g_proc, dtype=torch.float32)
        y_g_t = torch.tensor(y_test, dtype=torch.long)
        g_loader = DataLoader(TensorDataset(X_g_t, y_g_t), batch_size=config.BATCH_SIZE, shuffle=False)
        
        y_true, y_pred, y_prob = get_predictions_pytorch(model_g_nn, g_loader, device)
        all_results['FFT-Global-NN'] = {
            'metrics': calculate_metrics(y_true, y_pred, y_prob),
            'y_true': y_true,
            'y_prob': y_prob,
            'y_pred': y_pred
        }

    # 3. FFT Blockwise NN
    print("\n--- Evaluating FFT Blockwise NN ---")
    scaler_b = joblib.load(os.path.join(config.PREPROCESSOR_DIR, "blockwise_pca_scaler.pkl"))
    pca_b = joblib.load(os.path.join(config.PREPROCESSOR_DIR, "blockwise_pca_pca.pkl"))
    X_b_proc = pca_b.transform(scaler_b.transform(X_blockwise))
    
    model_b_nn = FFTNet(X_b_proc.shape[1], num_classes).to(device)
    b_nn_path = os.path.join(config.PYTORCH_MODEL_DIR, "fft_nn_blockwise_pca.pth")
    if os.path.exists(b_nn_path):
        model_b_nn.load_state_dict(torch.load(b_nn_path, map_location=device))
        
        X_b_t = torch.tensor(X_b_proc, dtype=torch.float32)
        y_b_t = torch.tensor(y_test, dtype=torch.long)
        b_loader = DataLoader(TensorDataset(X_b_t, y_b_t), batch_size=config.BATCH_SIZE, shuffle=False)
        
        y_true, y_pred, y_prob = get_predictions_pytorch(model_b_nn, b_loader, device)
        all_results['FFT-Blockwise-NN'] = {
            'metrics': calculate_metrics(y_true, y_pred, y_prob),
            'y_true': y_true,
            'y_prob': y_prob,
            'y_pred': y_pred
        }

    # 4. FFT Blockwise XGBoost
    print("\n--- Evaluating FFT Blockwise XGBoost ---")
    xgb_b_path = os.path.join(config.XGBOOST_MODEL_DIR, "xgboost_blockwise_pca.json")
    if os.path.exists(xgb_b_path):
        model_xgb = build_xgboost_model(num_classes, config.XGB_PARAMS)
        model_xgb.load_model(xgb_b_path)
        
        y_pred, y_prob = get_predictions_xgb(model_xgb, X_b_proc)
        all_results['FFT-Blockwise-XGBoost'] = {
            'metrics': calculate_metrics(y_test, y_pred, y_prob),
            'y_true': y_test,
            'y_prob': y_prob,
            'y_pred': y_pred
        }

    # Generate Visualization
    print("\nGenerating visualizations...")
    plot_roc_curves(all_results, class_names, "artifacts/results/roc_curves.png")
    for name, data in all_results.items():
        plot_confusion_matrix(data['y_true'], data['y_pred'], class_names, name, f"artifacts/results/cm_{name.lower().replace('-', '_')}.png")

    # Generate Report
    print("\nGenerating report...")
    report_lines = [
        "# Model Performance Comparison Report",
        "\n## Summary Metrics\n",
        "| Model | Accuracy | Balanced Acc | F1 (Macro) | F1 (Weighted) | MCC | Kappa | Log Loss |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    for name, data in all_results.items():
        m = data['metrics']
        line = f"| **{name}** | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {m['f1_macro']:.4f} | {m['f1_weighted']:.4f} | {m['mcc']:.4f} | {m['cohen_kappa']:.4f} | {m['log_loss']:.4f} |"
        report_lines.append(line)
        
    report_lines.append("\n## Visualizations")
    report_lines.append("\n### ROC Curves")
    report_lines.append("![ROC Curves](artifacts/results/roc_curves.png)")
    
    report_lines.append("\n### Confusion Matrices")
    for name in all_results.keys():
        safe_name = name.lower().replace('-', '_')
        report_lines.append(f"#### {name}")
        report_lines.append(f"![{name} Confusion Matrix](artifacts/results/cm_{safe_name}.png)")
        
    with open("comparison_report.md", "w") as f:
        f.write("\n".join(report_lines))
        
    print("\nComparison report generated: comparison_report.md")

if __name__ == "__main__":
    main()
