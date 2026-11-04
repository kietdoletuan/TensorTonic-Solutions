def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    new_v = []
    for state in range(len(values)):
        Qs = []
        for action in range(len(transitions[state])):
            reward = rewards[state][action]
            num_pstates = len(transitions[state][action])
            expected_future_value = sum([values[s_] * transitions[state][action][s_] for s_ in range(num_pstates)])
            Q = reward + gamma * expected_future_value
            Qs.append(Q)
            print(reward, num_pstates, expected_future_value, Q)
        new_v.append(max(Qs))
    return new_v
    pass