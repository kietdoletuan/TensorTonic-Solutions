import torch

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 probability tensor with the same shape as logits.
    """
    max_logit = torch.max(logits, dim=1, keepdim=True).values
    return torch.exp(logits-max_logit) / torch.sum(torch.exp(logits-max_logit), dim=1, keepdim=True)
    
    pass
