def per_priorities_and_weights(td_errors: list, alpha: float, beta: float, epsilon: float = 1e-6) -> tuple:
    """
    Returns (probs, is_weights), two float lists rounded to four decimals.
    """
    N = len(td_errors)
    
    p = [(abs(delta) + epsilon)**alpha for delta in td_errors]
    p_sum = sum(p)

    P = [pi / p_sum for pi in p]
    w = [(1/N * 1/Pi)**(beta) for Pi in P]

    w_max = max(w)
    w_norm = [wi / w_max for wi in w]

    return (P, w_norm)
