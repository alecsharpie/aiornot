import torchvision.transforms as transforms
import numpy as np

class FFT2D(object):
    """2D FFT of an image."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.fft.fft2(image)
        return image, label

def get_transforms():
    """Define the transformations"""
    return transforms.Compose([
    transforms.Resize((224, 224)),
    # RandomCrop(),
    # RandomVerticalFlip(),
    # RandomRotation(),
    # RandomBrightness(),
    # RandomContrast(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # calculated on imagenet
])
