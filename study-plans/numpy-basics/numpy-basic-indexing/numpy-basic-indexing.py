import numpy as np

def extract_subarray(arr: list, row_start: int, row_stop: int, col_start: int, col_stop: int) -> np.ndarray:
    """
    Returns the selected 2D subarray with dtype float64.
    """
    return np.asarray(arr, dtype=np.float64)[row_start:row_stop, col_start:col_stop]
