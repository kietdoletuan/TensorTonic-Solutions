import torch
import torch.nn as nn

def manual_train_step(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion: nn.Module, lr: float) -> float:
    """
    Returns the pre-update batch loss as a Python float.
    """
    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter -= lr * parameter.grad
    
    model.zero_grad()

    return loss.item()
    pass
