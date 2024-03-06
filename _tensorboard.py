from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "dataset/train/ants/0013035.jpg"
img_PIL = Image.open(image_path)
ima_array = np.array(img_PIL)

writer.add_image("test", ima_array, 1, dataformats='HWC')

for i in range(100):
     writer.add_scalar("y = x", 2*i, i)

writer.close()