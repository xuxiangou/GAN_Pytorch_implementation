import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class CityscapesDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size, train=True, transform=None):
        super(CityscapesDataset, self).__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.img = os.listdir(img_dir)
        self.transform = transform
        self.train = train

    def __getitem__(self, index):
        image = np.array(Image.open(self.img_dir + "/" + self.img[index]).convert("RGB"))
        # mask = Image.open(self.mask_dir + "/" + "_".join(self.img[index].split("_")[:-1]) + self.mask).convert("RGB")

        image, mask = image[:, 256:, :], image[:, :256, :]
        if self.train:
            augmentations = self.transform[0](image=image, image0=mask)
            image, mask = augmentations['image'], augmentations['image0']
            augmentations_input, augmentations_mask = self.transform[1](image=mask), self.transform[2](image=image)
            image, mask = augmentations_mask['image'], augmentations_input['image']
        else:
            mask = self.transform(image=mask)['image']
        return image, mask

    def __len__(self):
        return len(self.img)

    @staticmethod
    def random_crop(image, mask, crop_size=256):
        h_start, w_start = random.random(), random.random()
        h, w = image.size[0], image.size[0]
        y1 = int((h - crop_size) * h_start)
        y2 = y1 + crop_size
        x1 = int((w - crop_size) * w_start)
        x2 = x1 + crop_size
        image = image.crop((x1, y1, x2, y2))
        mask = mask.crop((x1, y1, x2, y2))
        return image, mask
