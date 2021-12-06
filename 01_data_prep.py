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

from my_loader import real_img_query_loader

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

best_prec1 = 0

def main():
    global args
    args = parser.parse_args()
    print(args)
    # create model
    print("=> creating model '{}'".format(args.arch))
    

    from models.vgg import VGG
    model = nn.DataParallel(VGG('VGG16', nclass=args.num_classes), device_ids=range(1))


    checkpoint = torch.load('./checkpoint/cifar10_vgg16_teacher.pth')
    model.load_state_dict(checkpoint)
    cudnn.benchmark = True

    # Data loading code
    real_img_val_loader = torch.utils.data.DataLoader(
        real_img_query_loader(transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    ## image extraction
    logits_extract(real_img_val_loader, model)



def logits_extract(real_img_val_loader, model):

    model.eval()

    x_labeled_array = []

    y_labeled_label = []
    y_labeled_logits = []

    print('*** Real Image Loading...')
    for it, (img) in enumerate(real_img_val_loader):

        print('==> Processed Images: ', (it+1)*args.batch_size)
        logits_query = model(img)
        softmax = nn.Softmax() # declare softmax function before it is used
        probs_query = softmax(logits_query) # get probabilities with softmax
        predict, tag = torch.max(probs_query, dim=1)

        x_labeled_array.append(img.data.cpu().numpy())
        y_labeled_label.append(tag.data.cpu().numpy())
        y_labeled_logits.append(probs_query.data.cpu().numpy())


    x_labeled_array = np.concatenate(x_labeled_array, axis=0)
    y_labeled_label = np.concatenate(y_labeled_label, axis=0)
    y_labeled_logits = np.concatenate(y_labeled_logits, axis=0)

    print('==> Dumping image data ...')

    outpath_pkl = './images/query/cifar10_query_label_'+str(len(x_labeled_array))+'.pkl' # active learning
    print(outpath_pkl)

    pickle.dump({'x_labeled_oris':x_labeled_array, 'y_labeled_label':y_labeled_label,'y_labeled_logits':y_labeled_logits},open(outpath_pkl,'wb'),-1)

if __name__ == '__main__':
    main()

