import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    
    return float(np.dot(x_array, y_array))