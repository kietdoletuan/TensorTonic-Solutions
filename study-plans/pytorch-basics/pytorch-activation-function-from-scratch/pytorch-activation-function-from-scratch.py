import torch

def activate(x: torch.Tensor, method: str = "relu") -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as x.
    """
    if method == "sigmoid":
        return torch.where(
            x >= 0,
            1 / (1 + torch.exp(-x)),
            torch.exp(x) / (1 + torch.exp(x))
        )
    elif method == "tanh":
        x2 = 2 * x
        sig = torch.where(
            x2 >= 0,
            1 / (1 + torch.exp(-x2)),
            torch.exp(x2) / (1 + torch.exp(x2))
        )
        return 2 * sig - 1
    elif method == "leaky_relu":
        return torch.where(x>0, x, 0.01*x)
    else:
        return torch.where(x > 0, x, 0)
    pass
