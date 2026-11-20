def replay_buffer_sample(capacity: int, transitions: list, sample_indices: list) -> tuple:
    """
    Returns (states, actions, rewards, next_states, dones), each a list in sample order.
    """
    buffer = [None] * capacity
    head = 0
    for transition in transitions:
        buffer[head] = transition

        if head == (capacity - 1):
            head = 0
            continue
        head += 1
        
    states, actions, rewards, next_states, dones = [], [], [], [], []
    
    for transition in [buffer[idx] for idx in sample_indices]:
        for i, info in enumerate([states, actions, rewards, next_states, dones]):
            if i == 2:
                info.append(float(transition[i]))
                continue
            info.append(transition[i])
    
    return (states, actions, rewards, next_states, dones)