import os
import numpy as np

import torch
import re
import torch.nn as nn

from skimage.transform import resize, rescale


def save(ckpt_dir: str, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict()
    }, os.path.join(ckpt_dir, f'model_epoch_{epoch}'))

def load(ckpt_dir: str, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch
    
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(f.split('_')[-1]))

    dict_model = torch.load(os.path.join(ckpt_dir, ckpt_lst[-1]))
    epoch = int(ckpt_lst[-1])

    return dict_model['net'], dict_model['optim'], epoch

def add_sampling(img, type='random', opts=None):
    if type == 'uniform':
        dy = opts[0]
        dx = opts[1]

        msk = np.zeros_like(img)
        msk[::dy, ::dx, :] = 1
        dst = img * msk
    
    elif type == 'random':
        prob = opts[0]

        msk = np.random.rand(img.shape[0], img.shape[1], img.shape[2])
        msk = (msk < prob).astype(np.float32)
        dst = img * msk

    elif type == 'gaussian':
        ly = np.linspace(-1, 1, img.shape[0])
        lx = np.linspace(-1, 1, img.shape[1])

        x, y = np.meshgrid(lx, ly)

        x0 = opts[0]
        y0 = opts[1]
        sgmx = opts[2]
        sgmy = opts[3]
        a = opts[4]

        gaus = a * np.exp(-((x - x0)**2/(2*sgmx**2) + (y - y0)**2/(2*sgmy**2)))
        gaus = gaus[:, :, np.newaxis]

        rnd = np.random.rand(img.shape[0], img.shape[1], img.shape[2])
        msk = (gaus > rnd).astype(np.float32)
        dst = img * msk

    return dst

def add_noise(img, type='random', opts=None):
    if type == 'random':
        sgm = opts[0]

        noise = (sgm / 255.0) * np.random.rand(img.shape[0], img.shape[1], img.shape[2])

        dst = img + noise
        dst = np.clip(dst, a_min=0, a_max=1)
    
    return dst

# import matplotlib.pyplot as plt
# path = 'G:\\내 드라이브\\resnet\\data\\BSR\\BSDS500\\data\\images\\train\\8143.jpg'
# img = plt.imread(path) / 255

# img = add_noise(img, type='random', opts=[200])
# plt.imshow(img)
# plt.show()