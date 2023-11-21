from typing import Dict
import torch


def metrics(confusion_matrix: torch.Tensor) -> dict[str, float]:
    # Compute metrics from the confusion matrix. format is: [true class, predicted class]
    if confusion_matrix.ndim != 2 or confusion_matrix.size(0) != confusion_matrix.size(1):
        raise ValueError("Confusion matrix must be a square matrix.")
    
    num_classes = confusion_matrix.size(0)
    result: Dict[str, float] = {}

    # Plain Accuracy
    total = torch.sum(confusion_matrix).float()
    result['acc'] = (torch.trace(confusion_matrix) / total).item() if total > 0 else 0.0

    # F1 scores
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = torch.sum(confusion_matrix[:, i]) - tp
        fn = torch.sum(confusion_matrix[i, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        result[f"f1({i})"] = f1.item()
    
    return result


def update_confusion_matrix(
        confusion_matrix: torch.Tensor,
        true_classes: torch.Tensor,
        predicted_classes: torch.Tensor,
        class_weights: Optional[torch.Tensor]=None) -> None:
    
    # Update the confusion matrix with the given true and predicted classes
    for t_i, p_i in zip(true_classes, predicted_classes):
        confusion_matrix[t_i, p_i] += 1
