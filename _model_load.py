from _model_save import *
# method1
model = torch.load("vgg16_method1.pth")

# method2
vgg16 = torchvision.models.vgg16(pretrained=False)
# model = torch.load("vgg16_method2.pth") //output the dict
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))  # equals method1

# trap
model = torch.load("tyy_method1.pth")
print(model)
