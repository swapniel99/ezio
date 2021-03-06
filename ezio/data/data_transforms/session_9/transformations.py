import albumentations as A
from albumentations.pytorch import ToTensor
import numpy as np

# Calculate mean and std of cifar-10 dataset
mean = [0.49139968, 0.48215841, 0.44653091]
std = [0.24703223, 0.24348513, 0.26158784]


def denormalize_image(img):
    img = img.numpy().astype(dtype=np.float32)
    for i in range(img.shape[0]):
        img[i] = (img[i] * std[i]) + mean[i]

    return np.transpose(img, (1, 2, 0))


def perform_transformations(train=False, is_numpy=False):
    # Initialize transforms list
    final_transforms = list()

    # Apply transformations only for training dataset
    if train:
        albumentation_transforms = [
            A.ShiftScaleRotate(rotate_limit=7, shift_limit = (0.1, 0.1), scale_limit=(0.9, 1.1)),
            A.HorizontalFlip(p=0.5),
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