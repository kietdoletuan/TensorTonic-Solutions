import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor with the same shape as x.
        """
        if self.training == True and self.p != 0:
            if self.p == 1:
                return torch.zeros(x.shape)
            
            mask = x.bernoulli((1 - self.p))
            y = (mask * x) / (1 - self.p)

            return y
        else:
            return x
        pass
