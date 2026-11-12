def sarsa_update(q_table: list, state: int, action: int, reward: float, next_state: int, next_action: int, alpha: float, gamma: float) -> list:
    """
    Returns a copied Q-table after one SARSA update.
    """
    error = reward + gamma * q_table[next_state][next_action] - q_table[state][action]
    q_table[state][action] = q_table[state][action] + alpha * error

    return q_table
    