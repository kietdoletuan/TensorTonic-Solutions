import numpy as np

def linear_combination(vectors: list, coefficients: list) -> np.ndarray:
    """
    Returns the weighted sum as a float64 vector.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)

    print(coefficients.shape, vectors.shape)
    return coefficients @ vectors 
        