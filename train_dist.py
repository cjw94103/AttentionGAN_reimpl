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

def setup(rank, world_size):
    adress = 'tcp://127.0.0.1:' + str(args.port_num)
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size, init_method=adress)

def cleanup():
    torch.distributed.destroy_process_group()

def main_worker(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")

    # init process group
    batch_size = int(args.batch_size / world_size)
    num_workers = int(args.num_workers / world_size)
    setup(rank, world_size)

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
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    # load model
    G_AB = ResnetGenerator_our(input_nc=args.channels, output_nc=None, ngf=args.ngf, n_blocks=args.n_blocks).to(rank)
    G_AB = torch.nn.parallel.DistributedDataParallel(G_AB, device_ids=[rank])
    
    G_BA = ResnetGenerator_our(input_nc=args.channels, output_nc=None, ngf=args.ngf, n_blocks=args.n_blocks).to(rank)
    G_BA = torch.nn.parallel.DistributedDataParallel(G_BA, device_ids=[rank])
    
    D_A = NLayerDiscriminator(input_nc=args.channels, ndf=args.ndf, n_layers=4, norm_layer=torch.nn.InstanceNorm2d).to(rank)
    D_A = torch.nn.parallel.DistributedDataParallel(D_A, device_ids=[rank])
    
    D_B = NLayerDiscriminator(input_nc=args.channels, ndf=args.ndf, n_layers=4, norm_layer=torch.nn.InstanceNorm2d).to(rank)
    D_B = torch.nn.parallel.DistributedDataParallel(D_B, device_ids=[rank])

    # get optimizer
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # train
    train(args, G_AB, G_BA, D_A, D_B, train_dataloader, optimizer_G, optimizer_D_A, optimizer_D_B)

    # clean process
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main_worker, nprocs=world_size, args=(world_size, ), join=True)
    
if __name__ == '__main__':
    main()