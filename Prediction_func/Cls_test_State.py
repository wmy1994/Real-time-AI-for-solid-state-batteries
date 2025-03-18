""" test neuron network performace
Author@Mingyang
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from Cls_utils import get_network, get_test_dataloader
from Cls_dataset_gene import Trans


if __name__ == '__main__':

    root = '_model_data/622_Cls_RGBdata-longcyc'

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('-root', type=str, default=root, help='data root path')
    parser.add_argument('-filenameTest', type=str, default='Val', help='file name for test data')
    parser.add_argument('-prefix', type=str, default='State_Cls_', help='prediction object: Battery state')
    args = parser.parse_args()

    net = get_network(args)

    root = args.root
    file_name_Test = args.filenameTest
    prefix = args.prefix
    net_name = args.net

    tss = Trans()  # transforms for test data
    test_loader = get_test_dataloader(
        root, file_name_Test, prefix,
        batch_size=args.b,
        num_workers=4,
        shuffle=False,
        transform=tss.transforms_Test(),
    )

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    # correct_1 = 0.0
    # correct_5 = 0.0
    correct = 0.0
    total = 0

    nf = open(root + '/' + (prefix + file_name_Test + '_dataset.txt'), 'r')
    bsize = args.b
    lines = nf.readlines()
    imgname = []

    for line in lines:
        word = line.split()
        imgname.append(word[0])
    nf.close()

    f = open(root + '/' + net_name + '_' + prefix + file_name_Test + '_result.txt', 'w')

    with torch.no_grad():
        for n_iter, (images, labels) in enumerate(test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

            if args.gpu:
                images = images.cuda()
                labels = labels.cuda()
                # print('GPU INFO.....')
                # print(torch.cuda.memory_summary(), end='')

            outputs = net(images)

            _, preds = outputs.max(1)  
            correct += preds.eq(labels).sum()

            for i in range(len(preds)):
                f.write(str(imgname[i+n_iter*args.b])+'\t') 
                Dvalue = 0
                if labels[i] == preds[i]:
                    Dvalue = 1
                f.write(str(int(labels[i])) + '\t' + str(int(preds[i])) + '\t' + str(Dvalue) + '\n')

    f.close()

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print('Test finished on txt file')
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
    print("Accuracy: ", float(correct / len(test_loader.dataset)))
    # print("Top 1 err: ", 1 - correct_1 / len(test_loader.dataset))
    # print("Top 5 err: ", 1 - correct_5 / len(test_loader.dataset))

