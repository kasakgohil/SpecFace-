from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch

def top_k_accuracy(outputs, labels, k=5):
    _, topk = outputs.topk(k, dim=1)
    correct = topk.eq(labels.view(-1, 1).expand_as(topk))
    return correct.sum().item()

def evaluate_pytorch_model(model, loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    total_top5 = 0
    preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)

            _, pred = outputs.max(1)
            total_correct += pred.eq(labels).sum().item()
            total_top5 += top_k_accuracy(outputs, labels, k=5)
            total_count += labels.size(0)
            
            preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_count if criterion else 0.0
    acc = total_correct / total_count
    top5_acc = total_top5 / total_count
    
    return avg_loss, acc, top5_acc, preds, all_labels

def print_evaluation_metrics(all_labels, preds, target_names=None):
    acc = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, preds, average='macro', zero_division=0)

    print("\n===== TEST METRICS =====")
    print(f"Top-1 Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    if target_names:
        print("\n===== CLASSIFICATION REPORT =====")
        print(classification_report(all_labels, preds, target_names=target_names))
        
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
