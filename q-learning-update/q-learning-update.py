import numpy as np

def q_learning_update(Q: list, s: int, a: int, r: float, s_next: int, alpha: float, gamma: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as Q.
    """
    Q = np.asarray(Q, dtype=np.float64)
     
    y = r + gamma * max([a_next for a_next in Q[s_next]])

    Q[s, a] += alpha * (y - Q[s, a])

    return Q
    
    