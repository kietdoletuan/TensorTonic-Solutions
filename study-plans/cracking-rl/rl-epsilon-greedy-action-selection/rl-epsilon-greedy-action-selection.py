import numpy as np

def epsilon_greedy_probs(Q_values: list, epsilon: float) -> list:
    """
    Returns the action-probability list, rounded to four decimals.
    """
    Q_values = np.asarray(Q_values, dtype=np.float64)
    best_a = np.argmax(Q_values)

    probs_list = [epsilon/len(Q_values) for a in Q_values]
    probs_list[best_a] += (1 - epsilon)

    return probs_list