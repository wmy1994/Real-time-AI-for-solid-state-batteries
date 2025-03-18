""" function
Author@Mingyang
"""
import os
import sys
import re
import datetime

import numpy
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Prediction_func.Cls_dataset_gene import BatteryDataset


def get_network_Cls(args):
    """ return given network
    """

    # modified classification network
    if args.net == 'resnet18':
        from models_cls.resnet_cls import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models_cls.resnet_cls import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models_cls.resnet_cls import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models_cls.resnet_cls import resnet101
        net = resnet101()
    elif args.net == 'inceptionv4':
        from models_cls.inceptionv4_cls import inceptionv4
        net = inceptionv4()
    elif args.net == 'densenet':
        from models_cls.densenet_cls import densenet121
        net = densenet121()
    elif args.net == 'mobilenetv2':
        from models_cls.mobilenetv2_cls import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'shufflenetv2':
        from models_cls.shufflenetv2_cls import shufflenetv2
        net = shufflenetv2()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu:  # use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(root, file_name, prefix, batch_size=16,
                            num_workers=2, shuffle=True, transform=None):
    """ return training dataloader
    Args:
        root: root path to dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle the data
        transform: whether to transform the img data: Trans.transforms_Train
    Returns: train_data_loader:torch dataloader object
    """

    dataset_train = BatteryDataset(root, file_name, prefix, transform)
    training_data_loader = DataLoader(
        dataset=dataset_train, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return training_data_loader


def get_test_dataloader(root, file_name, prefix, batch_size=16,
                            num_workers=2, shuffle=True, transform=None):
    """ return val or test dataloader
    Args:
        root: root path to dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: test_loader:torch dataloader object
    """
    dataset_test = BatteryDataset(root, file_name, prefix, transform)
    test_data_loader = DataLoader(
        dataset=dataset_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_data_loader


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimizer(e.g. SGD)
        total_iters: total_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]


def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]


def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch


def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]