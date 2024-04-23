import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import numpy as np
import itertools
import argparse

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import ImageDataset
from PIL import Image
from models import ResnetGenerator_our, NLayerDiscriminator
from make_args import Args
from train_func import train

from utils import *

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default='./config/apple2orange_single_gpu.json', help="config path")
opt = parser.parse_args()

# load config.json
args = Args(opt.config_path)

# make dataloader
train_transforms_ = [
    transforms.Resize(int(args.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((args.img_height, args.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

train_dataloader =  DataLoader(
    ImageDataset(args.data_path, transforms_=train_transforms_, unaligned=True, mode='train'),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers)

# load model
G_AB = ResnetGenerator_our(input_nc=args.channels, output_nc=None, ngf=args.ngf, n_blocks=args.n_blocks).to('cuda')
G_BA = ResnetGenerator_our(input_nc=args.channels, output_nc=None, ngf=args.ngf, n_blocks=args.n_blocks).to('cuda')
D_A = NLayerDiscriminator(input_nc=args.channels, ndf=args.ndf, n_layers=4, norm_layer=torch.nn.InstanceNorm2d).to('cuda')
D_B = NLayerDiscriminator(input_nc=args.channels, ndf=args.ndf, n_layers=4, norm_layer=torch.nn.InstanceNorm2d).to('cuda')

# optimizer
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# train
train(args, G_AB, G_BA, D_A, D_B, train_dataloader, optimizer_G, optimizer_D_A, optimizer_D_B)