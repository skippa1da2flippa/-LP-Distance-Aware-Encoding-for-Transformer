import torch.nn as nn
from typing import Type, Tuple
from torch import Tensor


class MaskedActivation(nn.Module):
    def __init__(self, f: Type[nn.Module] = nn.Sigmoid) -> None:
        """Initialize the MaskedActivation instance and store its configuration."""
        super().__init__()
        self.f: nn.Module = f()

    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """Run the forward pass for this module."""
        x_batch, masks = batch

        return self.f(x_batch), masks
