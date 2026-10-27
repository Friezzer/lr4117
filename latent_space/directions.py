from typing import Tuple, List
import numpy as np
from sklearn.svm import LinearSVC

def compute_difference_of_means(
    latent_z: np.ndarray, 
    confidence_scores: np.ndarray, 
    attr_idx: int, 
    top_percent: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = confidence_scores[:, attr_idx]
    n_top = max(1, int(len(scores) * top_percent))

    pos_indices = np.argsort(scores)[-n_top:]
    neg_indices = np.argsort(scores)[:n_top]

    z_bar_1 = np.mean(latent_z[pos_indices], axis=0)
    z_bar_0 = np.mean(latent_z[neg_indices], axis=0)

    direction = z_bar_1 - z_bar_0
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        raise ValueError("Вектор разности средних имеет нулевую норму.")
        
    direction_normalized = direction / norm
    return direction_normalized, z_bar_0, z_bar_1

def compute_svm_hyperplane_direction(
    latent_z: np.ndarray, 
    binary_labels: np.ndarray, 
    attr_idx: int
) -> Tuple[np.ndarray, LinearSVC]:
    y = binary_labels[:, attr_idx]
    clf = LinearSVC(C=1.0, max_iter=5000, random_state=42, dual=False)
    clf.fit(latent_z, y)
    
    w = clf.coef_[0]
    normal_vector = w / np.linalg.norm(w)
    return normal_vector, clf

def orthogonalize_gram_schmidt(
    target_direction: np.ndarray, 
    preserve_directions: List[np.ndarray]
) -> np.ndarray:
    v_ortho = target_direction.copy()
    for u in preserve_directions:
        u_norm = u / np.linalg.norm(u)
        proj = np.dot(v_ortho, u_norm) * u_norm
        v_ortho = v_ortho - proj

    norm = np.linalg.norm(v_ortho)
    if norm < 1e-8:
        raise ValueError("Коллинеарные векторы: проекция вырождена.")
    return v_ortho / norm