import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

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

class FFT2D(object):
    """2D FFT of an image."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.fft.fft2(image)
        return image, label
