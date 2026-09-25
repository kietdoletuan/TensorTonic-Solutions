def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    # Write code here
    old_values = values.copy()
    for state in range(len(transitions)):
        Qs = []
        for action in range(len(transitions[state])):
            sum = 0
            for v in range(len(old_values)):
                sum += (transitions[state][action][v] * old_values[v])
            Q = rewards[state][action] + gamma * sum
            Qs.append(Q)
        values[state] = max(Qs)

    return values
    
    pass