import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    if seq_len < 1 or d_model < 1:
        return []
    result = []

    half = int(np.ceil(d_model/2))
    
    for pos in range(seq_len):
        seq_pos_encode = []
        
        for i in range(half):
            print(i, half)
            pe_sin = float(np.sin(pos / base**(2*i / d_model)))
    
            if d_model % 2 != 0 and (i+1) == half:
                seq_pos_encode.append(pe_sin)
                continue
                
                
            pe_cos = float(np.cos(pos / base**(2*i / d_model)))
            
            seq_pos_encode.append(pe_sin)
            seq_pos_encode.append(pe_cos)
            
        result.append(seq_pos_encode)
    return np.asarray(result, dtype=float)
    pass