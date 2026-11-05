import numpy as np

def outer_sum(a: list, b: list) -> np.ndarray:
    """
    Returns an (m, n) float64 array of pairwise sums.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)

    a, b = a[:, np.newaxis], b[np.newaxis, :]
    print(a.shape, b.shape)

    return a + b
