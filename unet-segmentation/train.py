from cmath import log
from pprint import pformat
from random import Random
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
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./data", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

args = parser.parse_args()

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(pformat(args))

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'npy'))

if mode == "train":
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transforms=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transforms=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_train / batch_size)

else:
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transforms=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    num_data_test = len(dataset_test)
    
    num_batch_test = np.ceil(num_data_test / batch_size)

## Network
net = UNet().to(device)

optim = torch.optim.Adam(net.parameters(), lr=lr)

## additional functions
fn_loss = nn.BCEWithLogitsLoss().to(device)
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x*std) + mean
fn_class = lambda x: 1 * ( x > 0.5 )

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
            label_data = fn_tonumpy(label_data)
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image(
                'label',
                label_data,
                (num_batch_train * (epoch-1)) + batch,
                dataformats="NHWC"
            )
            writer_train.add_image(
                'input',
                input_data,
                (num_batch_train * (epoch-1)) + batch,
                dataformats="NHWC"
            )
            writer_train.add_image(
                'output',
                output,
                (num_batch_train * (epoch-1)) + batch,
                dataformats="NHWC"
            )

            print(f"batch : {batch} || loss : {np.mean(loss_arr)}")
        
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

                input_data = fn_tonumpy(input_data)
                label_data = fn_tonumpy(fn_denorm(label_data, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))

                writer_val.add_image(
                    'input',
                    input_data,
                    (num_batch_val * (num_epoch-1)) + batch,
                    dataformats='NHWC'
                )
                writer_val.add_image(
                    'label',
                    label_data,
                    (num_batch_val * (num_epoch-1)) + batch,
                    dataformats='NHWC'
                )
                writer_val.add_image(
                    'output',
                    output,
                    (num_batch_val * (num_epoch-1)) + batch,
                    dataformats='NHWC'
                )
        
        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
        
        if epoch % 50 == 0:
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
            label_data = fn_tonumpy(label_data)
            output = fn_tonumpy(fn_class(output))

            for j in range(label_data.shape[0]):
                id = num_batch_test * (batch -1) + j

                plt.imsave(os.path.join(result_dir, 'png', f'label_{id:.4f}.png'), label_data[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', f'input_{id:.4f}.png'), input_data[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', f'output_{id:.4f}.png'), output[j].squeeze(), cmap='gray')

                np.save(os.path.join(result_dir, 'npy', f'label_{id:.4f}.npy'), label_data[j].squeeze())
                np.save(os.path.join(result_dir, 'npy', f'input_{id:.4f}.npy'), input_data[j].squeeze())
                np.save(os.path.join(result_dir, 'npy', f'output_{id:.4f}.npy'), output[j].squeeze())
            
        print(f"Average test set : Loss : {np.mean(loss_arr)}")
            




















