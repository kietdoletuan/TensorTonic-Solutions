import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x = np.asarray(x)
    multiplier = 1 / (1 - p) 
    
    ran = rng.random(size=x.shape)
    mask = np.ones(x.shape) * multiplier
    mask[ran <= p] = 0

    return (x*mask, mask)
    
    pass