import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import pdb

import pickle
import numpy as np

from my_loader import active_query_loader

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Querying')

parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--num_classes',default=10, type=int, help='num of class in the model')
parser.add_argument('--dataset',default='cifar10',help='which dataset to train')


def main():
    global args
    args = parser.parse_args()
    print(args)
    # create model
    print("=> creating model '{}'".format(args.arch))
    
    if args.arch == 'vgg16':
        from models.vgg import VGG
        model = nn.DataParallel(VGG('VGG16', nclass=args.num_classes), device_ids=range(1))
    if args.arch == 'resnet18':
        from models.resnet import ResNet18
        model = nn.DataParallel(ResNet18().cuda())



    checkpoint = torch.load('./checkpoint/cifar10_vgg16_teacher.pth')
    model.load_state_dict(checkpoint)
    cudnn.benchmark = True

    # Data loading code

    mix_img_val_loader = torch.utils.data.DataLoader(
        active_query_loader(transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    ## image extraction
    logits_extract(mix_img_val_loader, model)


def logits_extract(mix_val_loader, model):

    model.eval()

    x_1_labeled_array = []
    x_2_labeled_array = []
    mix_array = []

    y_labeled_label = []
    y_labeled_logits = []
    x_1_unlabeled_array = []
    x_2_unlabeled_array = []

    mix_weights = []

    print('*** Image Mix Starts ...')
    for it, (img_mix) in enumerate(mix_val_loader):

        print('==> Processed 100 Images: ', (it+1))
        logits_query = model(img_mix)
        softmax = nn.Softmax()
        probs_query = softmax(logits_query*1) # get probabilities with softmax
        predict, tag = torch.max(probs_query, dim=1)

        mix_array.append(img_mix.data.cpu().numpy())
        y_labeled_label.append(tag.data.cpu().numpy())
        y_labeled_logits.append(probs_query.data.cpu().numpy())


    mix_array = np.concatenate(mix_array, axis=0)
    y_labeled_label = np.concatenate(y_labeled_label, axis=0)
    y_labeled_logits = np.concatenate(y_labeled_logits, axis=0)
    print('==> New Images: ', len(mix_array))

    #################################################################
    ############## Obtained Image Data for Concatenation ############
    datainfo = pickle.load(open('./images/query/cifar10_query_label_1000.pkl', 'rb'))
    x_labeled_oris_1 = datainfo['x_labeled_oris']
    y_labeled_label_old = datainfo['y_labeled_label']
    y_labeled_logits_old = datainfo['y_labeled_logits']
    print('==> Original Images: ', len(x_labeled_oris_1))

    ### Concatenate Obtained and New Data ###
    x_labeled_array = np.concatenate([mix_array, x_labeled_oris_1], axis=0)
    y_new_label = np.concatenate([y_labeled_label, y_labeled_label_old], axis=0)
    y_new_logits = np.concatenate([y_labeled_logits, y_labeled_logits_old], axis=0)

    print('==> Dump Query-labeled Data ...')
    outpath_pkl = './images/query/cifar10_query_label_'+str(len(x_labeled_array))+'.pkl' # active learning
    print(outpath_pkl)
    pickle.dump({'x_labeled_oris':x_labeled_array,'y_labeled_label':y_new_label,'y_labeled_logits':y_new_logits},open(outpath_pkl,'wb'),-1)


if __name__ == '__main__':
    main()





