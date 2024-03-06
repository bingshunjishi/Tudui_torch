import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root="./dataset", transform=torchvision.transforms.ToTensor(), train=False)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

writer = SummaryWriter("logs")
step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images("test_data", imgs, step)
    step = step + 1
writer.close()