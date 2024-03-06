import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([
    [1, -0.5],
    [-1, 3]
])
input_reshape = torch.reshape(input, (-1, 1, 2, 2))

dataset = torchvision.datasets.CIFAR10("dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class Tyy(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, x):
        output_relu1 = self.relu1(x)
        output_sigmoid1 = self.sigmoid1(x)
        return output_relu1, output_sigmoid1


tyy = Tyy()

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output_relu1, output_sigmoid1 = tyy(imgs)
    writer.add_images("output_relu", output_relu1, step)
    writer.add_images("output_sigmoid", output_sigmoid1, step)
    step += 1
writer.close()