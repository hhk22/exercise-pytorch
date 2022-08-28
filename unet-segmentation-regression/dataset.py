#!/usr/bin/env python
# coding: utf-8

# In[32]:


import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms

from util import *


# In[65]:


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms=None, task=None, opts=None):
        self.img_input_list, self.img_label_list = [], []
        self.data_dir = data_dir
        self.transforms = transforms
        self.task = task
        self.opts = opts

        filename_list = os.listdir(data_dir)
        filename_list = [f for f in filename_list if f.endswith('jpg') or f.endswith('png')]
        
        self.lst_data = filename_list
    
    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):
        img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
        img_size = img.shape
        
        if img_size[0] > img_size[1]:
            img = img.transpose([1, 0, 2]) # always long horizontal image

        if img.dtype == 'uint8':
            img = img / 255.0
        
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        
        label_image = img
        input_image = None
        if self.task == 'denoising':
            input_image = add_noise(img, type=self.opts[0], opts=self.opts[1])
        elif self.task == 'inpainting':
            input_image = add_sampling(img, type=self.opts[0], opts=self.opts[1])
        elif self.task == 'super_resolution':
            input_image = add_blur(img, type=self.opts[0], opts=self.opts[1])

        data = {
            'input': input_image,
            'label': label_image
        }

        if self.transforms:
            data = self.transforms(data)

        return data


# In[66]:


class RandomFlip():
    def __call__(self, data):
        img_input = data['input']
        img_label = data['label']

        if np.random.rand() > 0.5:
            img_input = np.fliplr(img_input)
            img_label = np.fliplr(img_label)
        if np.random.rand() > 0.5:
            img_input = np.flipud(img_input)
            img_label = np.flipud(img_label)

        return {
            'input': img_input,
            'label': img_label
        }


# In[67]:


class Normalization():
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
    
    def __call__(self, data):
        img_input = data['input']
        img_label = data['label']
        
        img_input = (img_input - self.mean) / self.std
        img_label = (img_label - self.mean) / self.std
        
        return {
            'input': img_input,
            'label': img_label
        }


# In[68]:


class ToTensor(object):
    def __call__(self, data):
        img_input = data['input']
        img_label = data['label']

        img_input = img_input.transpose((2, 0, 1)).astype(np.float32)
        img_label = img_label.transpose((2, 0, 1)).astype(np.float32)

        return {
            'input': torch.from_numpy(img_input),
            'label': torch.from_numpy(img_label)
        }


class RandomCrop(object):
    def __init__(self, shape):
        self.shape = shape
    
    def __call__(self, data):
        input_data, label_data = data['input'], data['label']

        h, w = input_data.shape[:2]
        new_h, new_w = self.shape

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top+new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, top+new_w, 1)

        input_data = input_data[id_y, id_x]
        label_data = label_data[id_y, id_x]

        return {
            'input': input_data, 
            'label': label_data
        }

