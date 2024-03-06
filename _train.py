import torch
from torch.utils.tensorboard import SummaryWriter

from _model import *
import torchvision
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root="dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# datasets' length
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f'train datasets\' length：{train_data_size}')
print(f'test datasets\' length：{test_data_size}')

# use DataLoader to load datasets
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# create the neural network model
tyy = Tyy()

# loss func
loss_fn = nn.CrossEntropyLoss()

# optimizer
leaning_rate = 0.01
optimizer = torch.optim.SGD(tyy.parameters(), lr=leaning_rate)

# set parameters of the neural network
total_train_step = 0  # train epoch
total_test_step = 0  # test epoch
epoch = 10  # train times

# 添加tensorboard
writer = SummaryWriter("logs")

for i in range(epoch):
    print(f'第{i + 1}轮训练开始')
    # train start
    # tyy.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = tyy(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f'训练次数：{{{total_train_step}}}, loss：{{{loss.item()}}}')
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    # test start
    # tyy.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = tyy(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print(f'整体测试集上的loss：{total_test_loss}')
    print(f'整体测试集上的正确率：{total_accuracy / test_data_size}')
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1
    torch.save(tyy, f"tyy_{i}.pth")
    print('模型已保存')
writer.close()
