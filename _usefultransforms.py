from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# PIL -> ToTensor
img = Image.open("dataset/train/ants/0013035.jpg")
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)

# writer
writer = SummaryWriter("logs")
writer.add_image("ToTensor", img_tensor)

# normalization
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5, ])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm)

# resize
trans_resize = transforms.Resize((512, 512))
'''img is a PIL'''
img_resize = trans_resize(img)
'''img is a Tensor'''
img_resize_tensor = trans_totensor(img_resize)
writer.add_image("Resize", img_resize_tensor)

# compose - resize - 2
trans_resize_2 = transforms.Resize(512)
'''trans_resize_2's output = trans_totensor's input'''
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize_compose", img_resize_2)

# randomcrop
trans_random = transforms.RandomCrop(512)
trans_compose = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose(img)
    writer.add_image("randomcrop", img_crop, i)
writer.close()
