import torch

def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as X.
    """
    mean = X.mean(dim=0)
    pvariation = ((X - mean)**2).mean(dim=0)

    y = gamma * ((X - mean) / (torch.sqrt(pvariation+eps))) + beta

    return y
    pass
