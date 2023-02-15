import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import numpy as np
import torch
from data.transform import get_transforms


class ImageDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.labels = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        label = self.labels.iloc[idx, 1]

        return image, label


def get_dataset():
    return ImageDataset(csv_file = '../raw_data/train.csv',
                                   root_dir = '../raw_data/train',
                                   transform = get_transforms())
