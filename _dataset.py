from torch.utils.data import Dataset
from PIL import Image
import os


class Mydata(Dataset):
    def __init__(self, root_dir, label_dir):
        self._root_dir = root_dir
        self._label_dir = label_dir
        self._path = os.path.join(self._root_dir, self._label_dir)
        self._img_path_list = os.listdir(self._path)

    def __getitem__(self, idx):
        img_name = self._img_path_list[idx]
        img_item_path = os.path.join(self._root_dir, self._label_dir, img_name)
        img = Image.open(img_item_path)
        label = self._label_dir
        return img, label

    def __len__(self):
        return len(self._img_path_list)


root_dir = "dataset/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = Mydata(root_dir, ants_label_dir)
bees_dataset = Mydata(root_dir, ants_label_dir)

train_dataset = ants_dataset + bees_dataset
