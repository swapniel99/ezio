import albumentations as A
from albumentations.pytorch import ToTensor
import numpy as np


def perform_transformations(train=False, is_numpy=False):
    # Initialize transforms list
    final_transforms = list()

    # Calculate mean and std of cifar-10 dataset
    mean = [0.49139968, 0.48215841, 0.44653091]
    std = [0.24703223, 0.24348513, 0.26158784]

    # Apply transformations only for training dataset
    if train:
        albumentation_transforms = [
            A.ShiftScaleRotate(rotate_limit=7, shift_limit = (0.1, 0.1), scale_limit=(0.9, 1.1)),
            A.HorizontalFlip(p=0.5),
            A.Cutout(max_holes=1, max_height=16, max_width=16, fill_value=mean*255)
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