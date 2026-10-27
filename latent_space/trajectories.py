from typing import List
import numpy as np

def slerp(val: float, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    low_norm = low / np.linalg.norm(low)
    high_norm = high / np.linalg.norm(high)
    dot = np.clip(np.dot(low_norm, high_norm), -1.0, 1.0)
    omega = np.arccos(dot)

    if np.isclose(omega, 0.0):
        return (1.0 - val) * low + val * high

    so = np.sin(omega)
    return (np.sin((1.0 - val) * omega) / so) * low + (np.sin(val * omega) / so) * high

def linear_interpolation(z_start: np.ndarray, z_end: np.ndarray, steps: int = 10) -> np.ndarray:
    alphas = np.linspace(0.0, 1.0, steps)
    return np.array([(1.0 - a) * z_start + a * z_end for a in alphas])

def spherical_interpolation(z_start: np.ndarray, z_end: np.ndarray, steps: int = 10) -> np.ndarray:
    alphas = np.linspace(0.0, 1.0, steps)
    return np.array([slerp(a, z_start, z_end) for a in alphas])

def bezier_curve_trajectory(
    z_start: np.ndarray, 
    z_control: np.ndarray, 
    z_end: np.ndarray, 
    steps: int = 10,
    project_to_sphere: bool = True
) -> np.ndarray:
    target_radius = np.sqrt(len(z_start))
    alphas = np.linspace(0.0, 1.0, steps)
    trajectory = []

    for t in alphas:
        b_t = (1.0 - t)**2 * z_start + 2.0 * (1.0 - t) * t * z_control + (t**2) * z_end
        if project_to_sphere:
            b_t = b_t * (target_radius / np.linalg.norm(b_t))
        trajectory.append(b_t)

    return np.array(trajectory)

def apply_semantic_edit(
    z_base: np.ndarray, 
    direction_v: np.ndarray, 
    alphas: List[float]
) -> np.ndarray:
    return np.array([z_base + a * direction_v for a in alphas])