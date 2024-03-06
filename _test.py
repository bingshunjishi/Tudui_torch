import torch
import torchvision.transforms.v2
from PIL import Image
from torch import nn

img_path = "imgs/dog.jpg"
image = Image.open(img_path)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])
image = transform(image)


class Tyy(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load("tyy_0.pth")
image = torch.reshape(image, (1, 3, 32, 32)).cuda()

model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1).item())
