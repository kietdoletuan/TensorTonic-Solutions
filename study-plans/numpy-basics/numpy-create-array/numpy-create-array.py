import numpy as np

def create_array(data: list) -> np.ndarray:
    """
    Returns a 2D float64 array with the same values and shape as data.
    """
    return np.asarray(data, dtype=np.float64)
    pass
