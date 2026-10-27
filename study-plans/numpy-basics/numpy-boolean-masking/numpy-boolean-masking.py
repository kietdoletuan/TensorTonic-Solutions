import numpy as np

def row_summary(data: list, threshold: float) -> np.ndarray:
    """
    Returns a float64 array of shape (3, m, n): mask, any-row, all-row.
    """
    data = np.asarray(data)
    mask = data > threshold

    element = mask.astype(np.float64)

    any_mask = mask.any(axis=1, keepdims=True)
    anyrow = np.where(any_mask, data, 0.0)

    all_mask = mask.all(axis=1, keepdims=True)
    allrow = np.where(all_mask, data, 0.0)

    return np.asarray([element, anyrow, allrow], dtype=np.float64)
    
    pass
