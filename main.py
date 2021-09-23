import torch
import torch.nn as nn
import os
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from datetime import datetime
import CNN
from train_model import train_model
from check_accuracy import check_accuracy
import argparse

# CUDA_VISIBLE_DEVICES=9 python3 main.py --lr 0.01 --batchsize 64

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a network')

    parser.add_argument('--lr', default=None, type=float,
                        help='learning rate of training')
    parser.add_argument('--batchsize', default=8, type=int,
                        help='batch size of training')
    parser.add_argument('--epoch', default=25, type=int,
                        help='training epochs')
    parser.add_argument('--download', default=False, type=bool,
                        help='data download')
    parser.add_argument('--numtrain', default=49000, type=int,
                        help='dataset used for training, 4900 is for test')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum of optimizer')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    BATCH_SIZE = args.batchsize
    n_epochs = args.epoch
    learning_rate = args.lr
    momentum = args.momentum
    tensorboard_root = '/data2/chengyi/assignment/AI_security_assignment1/mytensorboard'
    experiment_name = 'test_lr_' + str(args.lr) + '_batchsize_' + str(args.batchsize)
    tb_saved_path = os.path.join(tensorboard_root, datetime.now().strftime('%b%d_%H-%M-%S_') + experiment_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('using device:', device)

    transform_normal = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_aug = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    cifar10_train = dset.CIFAR10('/data2/chengyi/assignment/AI_security_assignment1', train=True,
                                 download=args.download, transform=transform_normal)
    loader_train = DataLoader(cifar10_train, batch_size=BATCH_SIZE,
                              sampler=sampler.SubsetRandomSampler(range(args.numtrain)))

    cifar10_val = dset.CIFAR10('/data2/chengyi/assignment/AI_security_assignment1', train=True,
                               download=args.download, transform=transform_normal)
    loader_val = DataLoader(cifar10_val, batch_size=BATCH_SIZE,
                            sampler=sampler.SubsetRandomSampler(range(49000, 50000)))

    cifar10_test = dset.CIFAR10('/data2/chengyi/assignment/AI_security_assignment1', train=False,
                                download=args.download, transform=transform_normal)
    loader_test = DataLoader(cifar10_test, batch_size=BATCH_SIZE)


    model = CNN.model(classnum=10)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
    trained_model = train_model(model, optimizer, loader_train, loader_val, device, tb_saved_path, n_epochs)
    check_accuracy(loader_test, trained_model, device=device)
