
from inspect import Parameter
import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms

from util import *
from dataset import CustomDataset, Normalization, RandomCrop, RandomFlip, ToTensor
from torch.utils.data import DataLoader
from model import *

parser = argparse.ArgumentParser(
    description = 'Train the UNet',
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--lr", default=1e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./data/BSR/BSDS500/data/images", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", choices=['on', 'off'], type=str, dest="train_continue")

parser.add_argument("--task", default="denoising", choices=["denoising", "inpainting"], type=str, dest="task")
parser.add_argument("--opts", nargs='+', default=["random", "90.0"], dest="opts")

parser.add_argument("--ny", default=320, type=int, dest="ny")
parser.add_argument("--nx", default=480, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nkernel", default=64, type=int, dest="nkernel")

parser.add_argument("--network", default="srresnet", choices=["srresnet", "resnet"], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")

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

task = args.task
opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float64)]

ny = args.ny
nx = args.nx
nch = args.nch
nker = args.nkernel

network = args.network
learning_type = args.learning_type

device = "cuda" if torch.cuda.is_available() else "cpu"

result_dir_train = os.path.join(result_dir, 'train')
result_dir_val = os.path.join(result_dir, 'val')
result_dir_test = os.path.join(result_dir, 'test')

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir_train, 'png'))
    os.makedirs(os.path.join(result_dir_val, 'png'))

    os.makedirs(os.path.join(result_dir_test, 'png'))
    os.makedirs(os.path.join(result_dir_test, 'npy'))

if mode == 'train':
    transform_train = transforms.Compose([RandomCrop(ny, nx), Normalization(), RandomFlip(), ToTensor()])
    transform_val = transforms.Compose([RandomCrop(ny, nx), Normalization(), RandomFlip(), ToTensor()])

    dataset_train = CustomDataset(os.path.join(data_dir, 'train'), transform=transform_train, task=task, opts=opts)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

    dataset_val = CustomDataset(os.path.join(data_dir, 'val'), transform=transform_val, task=task, opts=opts)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)

else:
    
    transform_test = transforms.Compose(RandomCrop(ny, nx), Normalization(), ToTensor())
    dataset_test = CustomDataset(os.path.join(data_dir, 'test'), transform=transform_test, task=task, opts=opts)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

net = None
if network == 'srresnet':
    net = SRResNet(in_channels=nch, out_channels=nch, nker=nker, learning_type=learning_type).to(device)
elif network == 'resnet':
    net = ResNet(in_channels=nch, out_channels=nch, nker=nker, learning_type=learning_type).to(device)

optim = torch.optim.Adam(net.parameters(), lr=lr)

fn_loss = nn.MSELoss().to(device)

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean=0.5, std=0.5: (x * std) + mean

cmap = None

st_epoch = 0
if mode == 'train':
    if train_continue == 'on':
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
    
    num_epoch = st_epoch + num_epoch
    
    for epoch in range(st_epoch, num_epoch):
        net.train()
        loss_mse = []

        for batch, data in enumerate(loader_train, 1):
            input_img = data['input'].to(device)
            label_img = data['label'].to(device)

            output_img = net(input_img)
            
            optim.zero_grad()
            
            loss = fn_loss(label_img, output_img)
            loss.backward()

            optim.step()

            loss_mse.append(loss.item())

            if batch % 10 == 0:
                print(f"TRAIN EPOCH:{epoch}, BATCH:{batch}, LOSS:{np.mean(loss_mse):.3f}")
                # label_img = fn_tonumpy(fn_denorm(label_img))
                # input_img = fn_tonumpy(fn_denorm(input_img))
                # output_img = fn_tonumpy(fn_denorm(output_img))
                label_img = fn_tonumpy(label_img)
                input_img = fn_tonumpy(input_img)
                output_img = fn_tonumpy(output_img)

                input_img = np.clip(input_img, a_min=0, a_max=1)
                label_img = np.clip(label_img, a_min=0, a_max=1)
                output_img = np.clip(output_img, a_min=0, a_max=1)
                
                id = (epoch*num_batch_train + batch)
                plt.imsave(os.path.join(result_dir_train, 'png', f'{id}_label.png'), label_img[0].squeeze(), cmap=cmap)
                plt.imsave(os.path.join(result_dir_train, 'png', f'{id}_input.png'), input_img[0].squeeze(), cmap=cmap)
                plt.imsave(os.path.join(result_dir_train, 'png', f'{id}_output.png'), output_img[0].squeeze(), cmap=cmap)
            
        
        with torch.no_grad():
            net.eval()
            loss_mse = []

            for batch, data in enumerate(loader_val, 1):
                label_img = data['label'].to(device)
                input_img = data['input'].to(device)
                
                output_img = net(input_img)

                loss = fn_loss(label_img, output_img)
                loss_mse.append(loss.item())
                print(f"VAL EPOCH:{epoch}, BATCH:{batch}, LOSS:{np.mean(loss_mse):.3f}")

                if batch % 10 == 0:
                    input_img = fn_tonumpy(input_img)
                    label_img = fn_tonumpy(label_img)
                    output_img = fn_tonumpy(output_img)

                    input_img = np.clip(input_img, a_min=0, a_max=1)
                    label_img = np.clip(label_img, a_min=0, a_max=1)
                    output_img = np.clip(output_img, a_min=0, a_max=1)

                    id = epoch * num_batch_val + batch
                    plt.imsave(os.path.join(result_dir_val, 'png', f'{id}_input.png'), input_img[0].squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(result_dir_val, 'png', f'{id}_label.png'), label_img[0].squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(result_dir_val, 'png', f'{id}_output.png'), output_img[0].squeeze(), cmap=cmap)

        if epoch % 20 == 0:
            print(f'model save, epoch:{epoch}, loss:{np.mean(loss_mse):.3f}')
            save(ckpt_dir, net, optim, epoch)

else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():
        net.eval()
        loss_mse = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            label_img = data['label'].to(device)
            input_img = data['input'].to(device)

            output_img = net(input_img)

            # 손실함수 계산하기
            loss = fn_loss(label_img, output_img)
            loss_mse += [loss.item()]

            print(f"TEST BATCH:{batch}, loss:{np.mean(loss_mse):.3f}")

            for j in range(label_img.shape[0]):
                id = batch_size * (batch - 1) + j

                label_ = label_img[j]
                input_ = input_img[j]
                output_ = output_img[j]

                np.save(os.path.join(result_dir_test, 'numpy', '%04d_label.npy' % id), label_)
                np.save(os.path.join(result_dir_test, 'numpy', '%04d_input.npy' % id), input_)
                np.save(os.path.join(result_dir_test, 'numpy', '%04d_output.npy' % id), output_)

                label_ = np.clip(label_, a_min=0, a_max=1)
                input_ = np.clip(input_, a_min=0, a_max=1)
                output_ = np.clip(output_, a_min=0, a_max=1)

                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_label.png' % id), label_, cmap=cmap)
                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_input.png' % id), input_, cmap=cmap)
                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_, cmap=cmap)

        print(f"AVERAGE TEST : LOSS: {np.mean(loss_mse):.3f}")

        


                




            















