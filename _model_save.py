import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

# method1: model structure + model parameters
torch.save(vgg16, "vgg16_method1.pth")

# method2: model parameters (smaller)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")


# trap
class Tyy(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 64, 3)

    def forward(self, x):
        x = self.conv1(x)
        return x


tyy = Tyy()
torch.save(tyy, "tyy_method1.pth")
