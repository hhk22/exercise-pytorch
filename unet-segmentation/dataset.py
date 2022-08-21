#!/usr/bin/env python
# coding: utf-8

# In[32]:


import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms


# In[65]:


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms=None):
        self.img_input_list, self.img_label_list = [], []
        self.data_dir = data_dir
        self.transforms = transforms

        filename_list = os.listdir(data_dir)
        for filename in filename_list:
            if filename.startswith('input'):
                self.img_input_list.append(filename)
            else:
                self.img_label_list.append(filename)
        
        self.img_input_list.sort()
        self.img_label_list.sort()
    
    def __len__(self):
        return len(self.img_input_list)

    def __getitem__(self, index):
        img_input = np.load(os.path.join(self.data_dir, self.img_input_list[index]))
        img_label = np.load(os.path.join(self.data_dir, self.img_label_list[index]))

        img_input = img_input / 255.0
        img_label = img_label / 255.0

        img_input = img_input[:, :, np.newaxis]
        img_label = img_label[:, :, np.newaxis]

        data = {
            'input': img_input,
            'label': img_label
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

