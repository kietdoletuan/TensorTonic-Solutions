import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    y = np.asarray(y, dtype=int)
    if len(y) > 0:
        _, counts = np.unique(y, return_counts=True)
        probability = counts / len(y)
        return float(-np.sum(probability * np.log2(probability)))
    else: 
        return 0.0
    pass