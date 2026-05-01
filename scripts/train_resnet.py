import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import config
from src.data.dataset import get_torchvision_datasets
from src.data.transforms import get_resnet_train_transforms, get_resnet_val_transforms
from src.models.resnet import build_resnet18
from src.training.trainer import train_model
from src.utils.metrics import evaluate_pytorch_model, print_evaluation_metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")
    
    model_path = os.path.join(config.PYTORCH_MODEL_DIR, "best_resnet18.pth")
    history_path = os.path.join(config.LOG_DIR, "resnet_baseline_log.csv")
    
    if os.path.exists(model_path) and os.path.exists(history_path):
        print(f"ResNet model already found at {model_path}. Skipping training and moving to next step...")
        return


    train_transform = get_resnet_train_transforms()
    val_transform = get_resnet_val_transforms()
    train_dataset, val_dataset, test_dataset = get_torchvision_datasets(config.DATASET_DIR, train_transform, val_transform)
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available())

    print(f"Building ResNet18 (Downloading weights if first time)...")
    model = build_resnet18(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.RESNET_LR)

    print("Training ResNet Baseline...")
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

    history_path = os.path.join(config.LOG_DIR, "resnet_baseline_log.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    
    model_path = os.path.join(config.PYTORCH_MODEL_DIR, "best_resnet18.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    print("Evaluating on Test Set...")
    _, test_acc, top5_acc, preds, all_labels = evaluate_pytorch_model(model, test_loader, device)
    print_evaluation_metrics(all_labels, preds, target_names=class_names)

if __name__ == "__main__":
    main()
