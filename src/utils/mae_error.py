import torch
import torch.nn as nn


class MAE_error(nn.Module):
    def __init__(self):
        super(MAE_error, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(output - target))
