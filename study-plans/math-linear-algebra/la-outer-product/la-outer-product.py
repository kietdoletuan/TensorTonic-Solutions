import numpy as np

def outer_product(u: list, v: list) -> np.ndarray:
    """
    Returns the float64 outer-product matrix.
    """
    result = []
    for i in range(len(u)):
        row = []
        for j in range(len(v)):
            row.append(u[i] * v[j])
        result.append(row)

    return np.asarray(result, dtype=np.float64)