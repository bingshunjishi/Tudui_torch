import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Tyy(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = Conv2d(3, 6, 3, 1, 0)

    def forward(self, x):
        r = self.conv1(x)
        return r


tyy = Tyy()
writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs, targets = data
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30])
    output = tyy(imgs)
    # torch.Size([64, 6, 30, 30]) -> [-1, 3, 30, 30]
    output_reshape = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output_reshape", output_reshape, step)
    step = step + 1
