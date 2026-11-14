import numpy as np


def cubic_r2(t, actual):
    """Residual and total sum of squares for a cubic fit pinned to the path's start and end."""
    start, end = actual[0], actual[-1]
    linear = start + t * (end - start)
    A = np.column_stack([t * (1 - t), t**2 * (1 - t)])
    coef, *_ = np.linalg.lstsq(A, actual - linear, rcond=None)
    fitted = linear + A @ coef
    return np.sum((actual - fitted) ** 2), np.sum((actual - actual.mean()) ** 2)
