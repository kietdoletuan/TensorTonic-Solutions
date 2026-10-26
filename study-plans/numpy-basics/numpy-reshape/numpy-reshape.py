import numpy as np

def reshape_array(data: list, operation: str) -> np.ndarray:
    """
    Returns a float64 array with the shape selected by operation.
    """
    data = np.asarray(data, dtype=np.float64)
    if operation == "flatten":
        return data.flatten()
    elif operation == "transpose":
        return data.T
    else:
        return np.expand_dims(data, axis=0)
    pass
