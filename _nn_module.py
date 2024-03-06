import torch
from torch import nn


class Tyy(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = input + 1
        return output


tyy = Tyy()
x = torch.tensor(1.0)
output = tyy(x)
print(output)
