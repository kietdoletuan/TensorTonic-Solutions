from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    x = np.asarray(x)
    mean = float(np.mean(x))
    median = float(np.median(x))
    values, counts = np.unique(x, return_counts=True)
    mode = float(values[counts.argmax()])

    return {
        'mean' : mean,
        'median' : median,
        'mode' : mode
    }
    pass