def dqn_loss(Q_online: list, Q_target_next: list, actions: list, rewards: list, dones: list, gamma: float) -> float:
    """
    Returns the mean squared TD loss, rounded to four decimals.
    """
    loss = 0
    for s in range(len(Q_online)):
        target = rewards[s] + gamma * (1 - dones[s]) * max(Q_target_next[s])
        delta = target - (Q_online[s][actions[s]])
        loss += (delta)**2
    return loss / len(Q_online)