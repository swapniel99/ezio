import albumentations as A
from albumentations.pytorch import ToTensor
import numpy as np
import cv2

# Calculate mean and std of cifar-10 dataset
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def perform_transformations(train=False, is_numpy=False):
    # Initialize transforms list
    final_transforms = list()

    # Apply transformations only for training dataset
    if train:
        albumentation_transforms = [
            A.PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_CONSTANT, value=np.array(mean)*255),
            A.RandomCrop(height=32, width=32, always_apply=True),
            A.HorizontalFlip(),
            A.Cutout(num_holes=1, max_h_size=16, max_w_size=16)
            ]
        final_transforms += albumentation_transforms

    # Normalize and convert to tensor
    final_transforms += [A.Normalize(mean=mean, std=std), ToTensor()]
    transforms = A.Compose(final_transforms)

    # Convert the transforms to numpy
    if is_numpy:
        return lambda img:transforms(image=np.array(img))["image"]
    else:
        return transforms