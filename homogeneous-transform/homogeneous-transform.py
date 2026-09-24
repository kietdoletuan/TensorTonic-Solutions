import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    T = np.asarray(T)
    ps = np.asarray(points)

    if T.shape == (4, 4) or ps.shape[1] == 3 or ps.shape[0] == 3:

        if len(ps.shape) == 2:
            results = []
            for p in ps:
                p = np.append(p, 1)
                transformed = T @ p
                result = transformed[:-1]
                results.append(result)
            return np.array(results)
            
        else:
            ps = np.append(ps, 1)
            transformed = T @ ps
            result = transformed[:-1]

            return result

        
    
    pass