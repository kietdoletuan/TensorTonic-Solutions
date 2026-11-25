def dueling_q_values(V: list, A_advantages: list) -> list:
    """
    Returns the Q-value matrix as a nested list, rounded to four decimals.
    """
    result = []
    for i in range(len(V)):
        mean = sum(A_advantages[i]) / len(A_advantages[i])
        centered_advantages = [Ai - mean for Ai in A_advantages[i]]
    
        Q = [cai + V[i] for cai in centered_advantages]
        result.append(Q)
    return result