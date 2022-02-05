import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image


class DataDimensionExpansion:
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, image, *args, **kwargs):
        transform = A.Compose(
            [
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )
        image = transform(image=image)['image']
        return image


class CelebADataset(Dataset):
    def __init__(self, data_dir, img_size):
        super(CelebADataset, self).__init__()
        self.data_dir = data_dir
        self.image = os.listdir(self.data_dir)
        self.img_size = img_size

    def __getitem__(self, index):
        image = np.array(Image.open(self.data_dir + '/' + self.image[index]).convert('RGB'))
        transform = DataDimensionExpansion(self.img_size)
        image = transform(image)

        return image

    def __len__(self):
        return len(self.image)
