import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
        )
        self.output =  nn.Linear(hidden_size, out_features)
        
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor of shape (batch, out_features).
        """
        x = self.hidden(x)
        x = self.output(x)
        return x
        
        pass
