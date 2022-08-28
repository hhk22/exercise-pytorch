from pprint import pformat
import numpy as np
import argparse
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import UNet
from dataset import *
from util import *

parser = argparse.ArgumentParser(
    description = 'Train the UNet',
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=50, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./data/BSR/BSDS500/data/images", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

parser.add_argument("--task", default="denoising", choices=["denoising", "inpainting", "super_resoltuion"], type=str, dest="task")
parser.add_argument("--opts", nargs='+', default=["random", "90.0"], dest="opts")

parser.add_argument("--ny", default=320, type=int, dest="ny")
parser.add_argument("--nx", default=480, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nkernel", default=64, type=int, dest="nkernel")

parser.add_argument("--network", default="unet", choices=["unet", "resnet", "autoencoder"], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")

args = parser.parse_args()
print(args)

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

task = args.task
opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float64)]

ny = args.ny
nx = args.nx
nch = args.nch
nkernel = args.nkernel

network = args.network
learning_type = args.learning_type

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(pformat(args))

result_dir_train = os.path.join(result_dir, 'train')
result_dir_val = os.path.join(result_dir, 'val')
result_dir_test = os.path.join(result_dir, 'test')
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir_train, 'png'))
    os.makedirs(os.path.join(result_dir_val, 'png'))

    os.makedirs(os.path.join(result_dir_test, 'png'))
    os.makedirs(os.path.join(result_dir_test, 'npy'))

if mode == "train":
    transform_train = transforms.Compose([RandomCrop(shape=(ny, nx)), Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
    transform_val = transforms.Compose([RandomCrop(shape=(ny, nx)), Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transforms=transform_train, task=task, opts=opts)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transforms=transform_val, task=task, opts=opts)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_train / batch_size)

else:
    transform_test = transforms.Compose([RandomCrop(shape=(ny, nx)), Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transforms=transform_test, task=task, opts=opts)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    num_data_test = len(dataset_test)
    
    num_batch_test = np.ceil(num_data_test / batch_size)

## Network
net = None
if network == 'unet':
    net = UNet(nch=nch, nker=nkernel, norm='bnorm', learning_type=learning_type).to(device)

optim = torch.optim.Adam(net.parameters(), lr=lr)

## additional functions
# fn_loss = nn.BCEWithLogitsLoss().to(device)
fn_loss = nn.MSELoss().to(device)

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x*std) + mean
# fn_class = lambda x: 1 * ( x > 0.5 )

## Summary Writer
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

st_epoch = 0
## Train Mode
if mode == 'train':
    if train_continue == 'on':
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch+1, num_epoch+1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train):
            input_data = data['input'].to(device)
            label_data = data['label'].to(device)

            output = net(input_data)

            optim.zero_grad()

            loss = fn_loss(output, label_data)
            loss.backward()

            optim.step()

            loss_arr.append(loss.item())

            input_data = fn_tonumpy(fn_denorm(input_data, mean=0.5, std=0.5))
            label_data = fn_tonumpy(fn_denorm(label_data, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            input_data = np.clip(input_data, a_min=0, a_max=1)
            output_data = np.clip(output, a_min=0, a_max=1)

            # id = int((num_batch_train * (epoch-1)) + batch)
            
            # plt.imsave(os.path.join(result_dir_train, 'png', f'{id:04d}_label.png'), label_data[0])
            # plt.imsave(os.path.join(result_dir_train, 'png', f'{id:04d}_input.png'), input_data[0])
            # plt.imsave(os.path.join(result_dir_train, 'png', f'{id:04d}_output.png'), output_data[0])

        print(f"epoch : {epoch} || loss : {np.mean(loss_arr)}")
        
        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)
    
        # Val Mode
        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val):
                input_data = data['input'].to(device)
                label_data = data['label'].to(device)
            
                output = net(input_data)

                loss = fn_loss(output, label_data)

                loss_arr.append(loss.item())

                input_data = fn_tonumpy(fn_denorm(input_data, mean=0.5, std=0.5))
                label_data = fn_tonumpy(fn_denorm(label_data, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

                id = int((num_batch_val * (epoch-1)) + batch)
                input_data = np.clip(input_data, a_min=0, a_max=1)
                output_data = np.clip(output, a_min=0, a_max=1)

                plt.imsave(os.path.join(result_dir_val, 'png', f'{id:04d}_input.png'), input_data[0])
                plt.imsave(os.path.join(result_dir_val, 'png', f'{id:04d}_label.png'), label_data[0])
                plt.imsave(os.path.join(result_dir_val, 'png', f'{id:04d}_output.png'), output_data[0])
        
        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
        
        if epoch % 50 == 0:
            print(f'model save | epoch : {epoch} | loss : {np.mean(loss_arr)}')
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)
        
    writer_train.close()
    writer_val.close()

else:
    net, optim, st_epoch = load(ckpt_dir, net, optim)

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            input_data = data['input'].to(device)
            label_data = data['label'].to(device)

            output = net(input_data)

            loss = fn_loss(output, input_data)

            loss_arr.append(loss.item())

            print(f"batch : {batch} || loss : {np.mean(loss_arr):.3f}")

            input_data = fn_tonumpy(fn_denorm(input_data, mean=0.5, std=0.5))
            label_data = fn_tonumpy(fn_denorm(label_data, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            for j in range(label_data.shape[0]):
                id = int(num_batch_test * (batch -1) + j)

                input_ = input_data[j]
                output_ = output[j]
                label_ = label_data[j]

                plt.imsave(os.path.join(result_dir_test, 'png', f'label_{id:04d}.png'), label_)
                plt.imsave(os.path.join(result_dir_test, 'png', f'input_{id:04d}.png'), input_)
                plt.imsave(os.path.join(result_dir_test, 'png', f'output_{id:04d}.png'), output_)

                np.save(os.path.join(result_dir_test, 'npy', f'label_{id:04d}.npy'), label_)
                np.save(os.path.join(result_dir_test, 'npy', f'input_{id:04d}.npy'), input_)
                np.save(os.path.join(result_dir_test, 'npy', f'output_{id:04d}.npy'), output_)
            
        print(f"Average test set : Loss : {np.mean(loss_arr)}")
            




















