#!/usr/bin/env python
# coding: utf-8

# In[1]:


## 라이브러리 추가하기
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# In[2]:


from torchvision import transforms, datasets


# In[4]:


lr = 1e-3
batch_size = 64
num_epoch = 10


# In[5]:


ckpt_dir = './checkpoint'
log_dir = './log'


# In[6]:


# Device Setting 
device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')


# In[7]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0, bias=True)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()
        
        self.fc1 = nn.Linear(in_features=320, out_features=50, bias=True)
        self.relu1_fc1 = nn.ReLU()
        self.drop1_fc1 = nn.Dropout2d(p=0.5)
        
        self.fc2 = nn.Linear(in_features=50, out_features=10, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.drop2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        
        x = x.view(-1, 320)
        
        x = self.fc1(x)
        x = self.relu1_fc1(x)
        x = self.drop1_fc1(x)
        
        x = self.fc2(x)
        
        return x


# In[8]:


def load(ckpt_dir, net, optim):
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort()

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])

    return net, optim


# In[9]:


# Minist Dataset Load
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
dataset = datasets.MNIST(download=True, root='./',  train=False, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# In[10]:


num_data = len(loader.dataset)
num_batch = np.ceil(num_data / batch_size)


# In[11]:


net = Net().to(device)
params = net.parameters()


# In[12]:


fn_loss = nn.CrossEntropyLoss().to(device)
fn_pred = lambda output: torch.softmax(output, dim=1)
fn_acc = lambda pred, label: ((pred.max(dim=1)[1] == label).type(torch.float)).mean()


# In[13]:


optim = torch.optim.Adam(params, lr=lr)
writer = SummaryWriter(log_dir=log_dir)


# In[14]:


net, optim = load(ckpt_dir, net, optim)


# In[16]:


with torch.no_grad():
    net.eval()
    
    loss_arr = []
    acc_arr = []
    
    for batch, (input_d, label) in enumerate(loader, 1):
        input_d = input_d.to(device)
        label = label.to(device)
        
        output = net(input_d)
        pred = fn_pred(output)
                
        loss = fn_loss(output, label)
        acc = fn_acc(pred, label)
        
        loss_arr.append(loss.item())
        acc_arr.append(acc.item())
        
        print('TEST: BATCH %04d/%04d | LOSS: %.4f | ACC %.4f' %
              (batch, num_batch, np.mean(loss_arr), np.mean(acc_arr)))  


# In[ ]:




