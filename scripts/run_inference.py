import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import xgboost as xgb
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import config
from src.data.dataset import get_torchvision_datasets
from src.data.transforms import get_resnet_train_transforms, get_resnet_val_transforms
from src.models.resnet import build_resnet18
from src.utils.metrics import evaluate_pytorch_model, print_evaluation_metrics
from sklearn.metrics import accuracy_score, classification_report

def main():
    parser = argparse.ArgumentParser(description="Run Inference on Saved Models")
    parser.add_argument("--model-type", choices=["resnet", "xgb_global", "nn_global"], required=True, help="Which model to test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test dataset
    transform = get_resnet_val_transforms()
    _, _, test_dataset = get_torchvision_datasets(config.DATASET_DIR, get_resnet_train_transforms(), transform)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    class_names = test_dataset.classes

    if args.model_type == "resnet":
        print("Loading ResNet...")
        model = build_resnet18(len(class_names)).to(device)
        model_path = os.path.join(config.PYTORCH_MODEL_DIR, "best_resnet18.pth")
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}. Please train it first.")
            return

        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        model.load_state_dict(checkpoint)
        
        print("Evaluating ResNet on Test Set...")
        _, test_acc, top5_acc, preds, all_labels = evaluate_pytorch_model(model, test_loader, device)
        print_evaluation_metrics(all_labels, preds, target_names=class_names)
    else:
        print("For FFT models, testing is built into the end of their respective train scripts.")
        print("You can extend this script to load FFT models and scalers dynamically.")

if __name__ == "__main__":
    main()
