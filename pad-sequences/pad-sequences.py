import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """

    if not seqs:
        return np.array([], dtype=int).reshape(0,0)
    
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    
    

    result = []

    for seq in seqs:
        if len(seq) > max_len:
            result.append(seq[:max_len])
        elif len(seq) < max_len:
            padding = [pad_value] * (max_len - len(seq))
            print(seq, padding)
            seq.extend(padding)
            result.append(seq)
        else:
            result.append(seq)
            
    print(result)
    return np.asarray(result, dtype=int)
    pass