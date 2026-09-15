import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    A = np.array(A)
    r, c = A.shape
    T = np.zeros((c, r), dtype=A.dtype)

    for i in range(r):
        for j in range(c):
            T[j, i] = A[i, j] 

    return T
    
    pass
