import torch
import torch.nn as nn
from tqdm import tqdm
from .callbacks import EarlyStopping

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, patience, device):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    history = []

    for epoch in range(epochs):
        # TRAIN
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]", leave=False)

        for images, labels in train_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)

            _, preds = outputs.max(1)
            train_total += labels.size(0)
            train_correct += preds.eq(labels).sum().item()

            train_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * train_correct / train_total:.2f}%"
            })

        train_acc = train_correct / train_total
        train_loss_epoch = train_loss / train_total

        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [VAL]", leave=False)

        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)

                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

                val_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100 * val_correct / val_total:.2f}%"
                })

        val_acc = val_correct / val_total
        val_loss_epoch = val_loss / val_total

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss_epoch,
            "train_acc": train_acc,
            "val_loss": val_loss_epoch,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        print(f"Epoch {epoch+1:03d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

    return model, history, early_stopping.val_acc_max
