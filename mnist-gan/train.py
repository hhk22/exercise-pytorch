
import os

import torch
import torch.nn as nn

import numpy as np
import argparse
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import Discriminator, Generator
from utils import save

parser = argparse.ArgumentParser(description="Gan Task with Mnist data",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_dir", default="./data", type=str, dest="data_dir")
parser.add_argument("--result_dir", default="./result_dir", type=str, dest="result_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")

args = parser.parse_args()

ckpt_dir = args.ckpt_dir
result_dir = args.result_dir
data_dir = args.data_dir

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

def set_requires_grad(net: nn.Module, requires_grad=False) -> None:
    for param in net.parameters():
        param.requires_grad = requires_grad

batch_size = 32

dataloader = DataLoader(
    datasets.MNIST(
        data_dir,
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ),
    batch_size=batch_size,
    shuffle=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 1e-3
num_epoch = 100
fn_loss = nn.BCELoss().to(device)
fn_tonumpy = lambda x: x.to('cpu').detach().numpy()

model_G = Generator().to(device)
model_D = Discriminator().to(device)

optim_G = torch.optim.Adam(model_G.parameters(), lr=lr, betas=(0.5, 0.999))
optim_D = torch.optim.Adam(model_D.parameters(), lr=lr, betas=(0.5, 0.999))

ones = torch.ones(batch_size, 1).to(device)
zeros = torch.zeros(batch_size, 1).to(device)

loss_D_arr = []
loss_G_arr = []

for epoch in range(num_epoch):
    output = None
    imgs = None
    for batch, (imgs, _) in enumerate(dataloader, 1):
        model_G.train()
        model_D.train()
        
        # Update Discriminator
        rand_vec = torch.from_numpy(np.random.rand(batch_size, 100).astype(np.float32)).to(device)
        
        output = model_G(rand_vec)

        imgs = imgs.squeeze()
        imgs = imgs.reshape(-1, 28*28)
        imgs = imgs.to(device)

        set_requires_grad(model_D, requires_grad=True)
        optim_D.zero_grad()
        
        pred_real = model_D(imgs)
        # detach : https://redstarhong.tistory.com/64
        pred_fake = model_D(output.detach()) # model_G : requires_grad = False

        loss_D_real = fn_loss(pred_real, ones)
        loss_D_fake = fn_loss(pred_fake, zeros)
        loss_D = (loss_D_real + loss_D_fake) / 2

        loss_D.backward()
        optim_D.step()

        set_requires_grad(model_D, requires_grad=False)
        optim_G.zero_grad()
        
        # Update Generator
        pred_fake = model_D(output)
        loss_G = fn_loss(pred_fake, ones)
        
        loss_G.backward()
        optim_G.step()

        loss_G_arr.append(loss_G.item())
        loss_D_arr.append(loss_D.item())

        if batch % 100 == 0:
            print(f'epoch: {epoch}, loss_G:{np.mean(loss_G_arr)}, loss_D:{np.mean(loss_D_arr)}')
    
    id = epoch

    output = fn_tonumpy(output[0]).reshape(28, 28)
    img = fn_tonumpy(imgs[0]).reshape(28, 28)

    plt.imsave(os.path.join(result_dir, f'{id}_output.png'), output.squeeze(), cmap=None)
    plt.imsave(os.path.join(result_dir, f'{id}_label.png'), img.squeeze(), cmap=None)
    save(ckpt_dir, model_G, model_D, optim_G, optim_D, epoch)






    

    




    


