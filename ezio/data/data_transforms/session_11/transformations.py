import albumentations as A
from albumentations.pytorch import ToTensor
import numpy as np
import cv2

# Calculate mean and std of cifar-10 dataset
mean = [0.49139968, 0.48215841, 0.44653091]
std = [0.24703223, 0.24348513, 0.26158784]


def perform_transformations(train=False, is_numpy=False):
    # Initialize transforms list
    final_transforms = list()

    # Apply transformations only for training dataset
    if train:
        albumentation_transforms = [
            A.PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_CONSTANT, value=np.array(mean)*255),
            A.RandomCrop(height=32, width=32, always_apply=True),
            A.HorizontalFlip(),
            A.Cutout(num_holes=1, max_h_size=8, max_w_size=8)
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