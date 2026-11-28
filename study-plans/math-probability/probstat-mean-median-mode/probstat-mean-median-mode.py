import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns mean, median, and mode as Python floats in a dictionary.
    """
    x = np.asarray(x, dtype=np.float64)
    vals, counts = np.unique(x, return_counts=True)
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "mode": vals[np.argmax(counts)]}