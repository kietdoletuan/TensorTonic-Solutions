import numpy as np

def original_and_clipped(data: list, row_idx: int, lo: float, hi: float) -> np.ndarray:
    """
    Returns a (2, n) float64 array: original row, then clipped row.
    """
    data = np.asarray(data, dtype=np.float64)

    lo_mask = data[row_idx] < lo
    hi_mask = data[row_idx] > hi
    
    clipped = data[row_idx].copy()
    clipped[lo_mask] = lo
    clipped[hi_mask] = hi

    return np.asarray([data[row_idx], clipped], dtype=np.float64)

    
