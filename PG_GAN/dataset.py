import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import albumentations as A


class Augmentation:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, image):
        augmentation = A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), max_pixel_value=255),
                ToTensorV2(),
            ]
        )

        return augmentation(image=image)['image']


class PGDataset(Dataset):
    def __init__(self, image_dir, image_size):
        super(PGDataset, self).__init__()
        self.image_dir = image_dir
        self.image_size = image_size
        self.image = os.listdir(self.image_dir)

    def __getitem__(self, index):
        image = np.array(Image.open(self.image_dir + '/' + self.image[index]).convert('RGB'))
        augmentation = Augmentation(self.image_size)
        image = augmentation(image)

        return image

    def __len__(self):
        return len(self.image)
