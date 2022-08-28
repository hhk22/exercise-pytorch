import os
import numpy as np

import torch
import torch.nn as nn

from skimage.transform import rescale, resize

## 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

def add_sampling(img, type='random', opts=None):
    img_size = img.shape

    if type == 'uniform':
        dy = opts[0]
        dx = opts[1]
        assert dy and dx, 'dy & dx is None'

        msk = np.zeros(img_size)
        msk[::dy, ::dx, :] = 1
        
        dst = img * msk
    
    elif type == 'random':
        prob = opts[0]
        assert prob, 'prob is None'

        rand = np.random.rand(img_size[0], img_size[1], img_size[2])
        msk = (prob > rand).astype(np.float64)
        dst = img * msk
    
    elif type == 'gaussian':
        ly = np.linspace(-1, 1, img_size[0])
        lx = np.linspace(-1, 1, img_size[1])
        x, y = np.meshgrid(lx, ly)

        x0 = opts[0]
        y0 = opts[1]
        sgmx = opts[2]
        sgmy = opts[3]
        a = opts[4]        

        assert x0!=None and y0!=None and sgmx and sgmy and a, 'gaussian variable is None'

        gaus = a * np.exp(-((x - x0)**2/(2*sgmx**2) + (y - y0)**2/(2*sgmy**2)))
        gaus = gaus[:, :, np.newaxis]

        rnd = np.random.rand(img_size[0], img_size[1], img_size[2])
        msk = (rnd < gaus).astype(np.float64)

        dst = img * msk
    
    return dst

def add_noise(img, type='random', opts=None):
    img_size = img.shape

    if type == 'random':
        sgm = opts[0]
        assert sgm != None, 'sgm variable is None'

        noise = (sgm / 255.0) * \
                np.random.rand(img_size[0], img_size[1], img_size[2])
        
        dst = img + noise

    return dst

def add_blur(img, type='bilinear', opts=None):
    order = None # default bilinear
    if type == 'nearest': order = 0
    elif type == 'bilinear': order = 1
    elif type == 'biquadratic': order = 2
    elif type == 'bicubic': order = 3
    elif type == 'biquartic': order = 4
    elif type == 'biquintic': order = 5

    img_size = img.shape
    keep_dim = True if opts[0] else False
    dw = opts[1]
    assert dw, 'variable dw is None'
    assert order!=None, 'variable order is None'

    dst = resize(
        img, 
        output_shape=(img_size[0]//dw, img_size[1]//dw, img_size[2]),
        order = order
    )
    if keep_dim:
        dst = resize(
            dst,
            output_shape=(img_size[0], img_size[1], img_size[2]),
            order = order
        )
    
    return dst


# import matplotlib.pyplot as plt
# img_path = './data/BSR/BSDS500/data/images/train/2092.jpg'
# img = plt.imread(img_path)
# img = img / 255.0

# img = add_noise(img, type='random', opts=[90])
# img = np.clip(img, a_min=0, a_max=1)
# plt.imshow(img)
# plt.show()


