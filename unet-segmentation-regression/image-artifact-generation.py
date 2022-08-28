import numpy as np
import os
import matplotlib.pyplot as plt

from skimage.transform import rescale, resize

img_path = './BSR/BSDS500/data/images/train/2092.jpg'

img = plt.imread(img_path)
img = img / 255.0

## uniform sampling 
def uniform_sampling(img):
    img_size = img.shape
    dy = 2
    dx = 2

    msk = np.zeros(img_size)
    msk[::dy, ::dx, :] = 1

    dst = img * msk
    return dst

def random_sampling(img):
    prob = 0.5
    img_size = img.shape
    rnd = np.random.rand(img_size[0], img_size[1], img_size[2])
    msk = (rnd > prob).astype(np.float64)

    dst = img * msk
    return dst

def gaussian_sampling(img):
    img_size = img.shape
    ly = np.linspace(-1, 1, img_size[0])
    lx = np.linspace(-1, 1, img_size[1])
    x, y = np.meshgrid(lx, ly)

    x0 = 0
    y0 = 0
    sgmx = 1
    sgmy = 1

    a = 1

    gaus = a * np.exp(-((x - x0)**2/(2*sgmx**2) + (y - y0)**2/(2*sgmy**2)))
    gaus = gaus[:, :, np.newaxis]

    rnd = np.random.rand(img_size[0], img_size[1], 1)
    msk = (rnd < gaus).astype(np.float64)

    dst = img * msk    
    return dst

def random_noising(img):
    img_size = img.shape
    sgm = 60.0

    noise = (sgm / 255.0) * np.random.rand(img_size[0], img_size[1], img_size[2])
    dst = img + noise

    return dst



img = random_noising(img)
plt.imshow(img)
plt.show()
