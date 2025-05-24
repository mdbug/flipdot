import numpy as np

def disolve(dots: np.array, alpha: float) -> np.array:
    # Ensure alpha is between 0 and 1
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("Alpha must be between 0.0 and 1.0")
    
    # Delete random pixels
    mask = np.random.rand(*dots.shape) < alpha
    dots[mask] = 0
    return dots

def resolve(dots: np.array, alpha: float) -> np.array:
    return disolve(dots, 1 - alpha)