from typing import Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_classification_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    attribute_names: List[str]
) -> Dict[str, Dict[str, float]]:
    metrics = {}
    f1_list, acc_list = [], []

    for idx, name in enumerate(attribute_names):
        acc = float(accuracy_score(y_true[:, idx], y_pred[:, idx]))
        prec = float(precision_score(y_true[:, idx], y_pred[:, idx], zero_division=0))
        rec = float(recall_score(y_true[:, idx], y_pred[:, idx], zero_division=0))
        f1 = float(f1_score(y_true[:, idx], y_pred[:, idx], zero_division=0))

        metrics[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        acc_list.append(acc)
        f1_list.append(f1)

    metrics["macro_avg"] = {
        "accuracy": float(np.mean(acc_list)),
        "f1": float(np.mean(f1_list))
    }
    return metrics

def compute_attribute_correlation_matrix(directions: List[np.ndarray]) -> np.ndarray:
    n = len(directions)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            u = directions[i] / np.linalg.norm(directions[i])
            v = directions[j] / np.linalg.norm(directions[j])
            corr_matrix[i, j] = float(np.dot(u, v))
    return corr_matrix