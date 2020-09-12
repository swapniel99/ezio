import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def show_random_images(dataset, classes):

	dataiter = iter(dataset)
	images, labels = dataiter.next()

	img_list = range(5, 10)
	# show images
	print('shape:', images.shape)
	imshow(torchvision.utils.make_grid(images[img_list]))