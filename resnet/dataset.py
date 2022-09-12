from typing import Dict
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision.transforms import Compose
from util import *

class CustomDataset(Dataset):
    def __init__(self, data_dir: str, transform: Compose = None, task=None, opts=None) -> None:
        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        self.opts = opts

        filenames = os.listdir(data_dir)
        self.lst_data = [filename for filename in filenames if filename.endswith('jpg') or filename.endswith('png')]
    
    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index) -> Dict:

        img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
        img_size = img.shape

        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        if img_size[0] > img_size[1]:
            img = img.transpose([1, 0, 2])
        if img.dtype == "uint8": # range : 0 ~ 255
            img = img / 255.0 # range : 0 ~ 1
        
        label_image = img
        input_image = None

        if self.task == 'denoising':
            input_image = add_noise(label_image, type=self.opts[0], opts=self.opts[1])
        elif self.task == 'inpainting':
            input_image = add_sampling(label_image, type=self.opts[0], opts=self.opts[1])
        
        data = {
            "input": input_image,
            "label": label_image
        }

        if self.transform:
            data = self.transform(data)
        
        return data

class RandomFlip():
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, data):
        input_image = data['input']
        label_image = data['label']

        if np.random.rand() < self.prob:
            input_image = np.fliplr(input_image)
            label_image = np.fliplr(label_image)
        if np.random.rand() < self.prob:
            input_image = np.flipud(input_image)
            label_image = np.flipud(label_image)
        
        return {
            'input': input_image,
            'label': label_image
        }

class Normalization():
    def __init__(self, mean=0.5, std=0.5) -> None:
        self.mean = mean
        self.std = std
    
    def __call__(self, data):
        input_image = data['input']
        label_image = data['label']

        input_image = (input_image - self.mean) / self.std
        label_image = (label_image - self.mean) / self.std

        return {
            'input': input_image,
            'label': label_image
        }

class RandomCrop():
    def __init__(self, ry, rx) -> None:
        self.shape = (ry, rx)
    
    def __call__(self, data):
        input_image = data['input']
        label_image = data['label']

        h, w, _ = input_image.shape
        new_h, new_w = self.shape

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top+new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, left+new_w, 1)

        return {
            'input': input_image[id_y, id_x],
            'label': label_image[id_y, id_x]
        }

class ToTensor():
    def __call__(self, data):
        input_image = data['input']
        label_image = data['label']

        input_image = input_image.transpose((2, 0, 1)).astype(np.float32)
        label_image = label_image.transpose((2, 0, 1)).astype(np.float32)

        input_image = torch.from_numpy(input_image)
        label_image = torch.from_numpy(label_image)

        return {
            'input': input_image,
            'label': label_image
        }



        


        
        
            
    







        