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

from my_loader import mix_img_query_loader
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
parser.add_argument('-b', '--batch-size', default=1000, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--num_classes',default=10, type=int, help='num of class in the model')
parser.add_argument('--dataset',default='cifar10',help='which dataset to train')

def main():
    global args
    args = parser.parse_args()
    print(args)
    # create model
    print("=> creating model '{}'".format(args.arch))
    print(torch.cuda.device_count())    
    if args.arch == 'vgg16':
        from models.vgg import VGG
        model = nn.DataParallel(VGG('VGG16', nclass=args.num_classes))
    elif args.arch == 'resnet18':
        from models.resnet import ResNet18
        model = nn.DataParallel(ResNet18().cuda())
    else:
        raise NotImplementedError('Invalid model')
    

############################################################
############# Modify Current Student Model #################

    checkpoint = torch.load('./active_student_models/cifar10_vgg_student_model_1000.pth')
    model.load_state_dict(checkpoint)
############################################################
    cudnn.benchmark = True
    model.cuda()

    # Data loading code
    mix_img_val_loader = torch.utils.data.DataLoader(
        mix_img_query_loader(transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    ## image extraction
    logits_extract(mix_img_val_loader, model)

def logits_extract(val_loader, model):
    model.eval()

    x_oris_1 = []
    x_oris_2 = []

    x_oris_1_array = []
    x_oris_2_array = []
    mix_image = []

    mix_images_weights = []
    lab = []
    unc_tags = []

    w = np.arange(0.30, 0.70+0.04, 0.04)

    for it, (img_1, img_2, img_1_path, img_2_path) in enumerate(val_loader):

        print('==> Processed K Images: ', (it+1))
        img_1, img_2 = img_1.cuda(), img_2.cuda()

        uncertainty = []
        best_w = []
        labels = []

        x_oris_1.extend(img_1_path)
        x_oris_2.extend(img_2_path)

        x_oris_1_array.append(img_1.data.cpu().numpy())
        x_oris_2_array.append(img_2.data.cpu().numpy())

###################### UNCERTAINTY POLICY #################
        for i in range(len(w)):
            mix = img_1*w[i] + img_2*(1-w[i]) # image merge; convex combination
            with torch.no_grad():
                logits_query = model(mix) # query teacher network to obtain logits
                softmax = nn.Softmax() # declare softmax function before it is used
            probs_query = softmax(logits_query) # get probabilities with softmax
            predict, tag = torch.max(probs_query, dim=1) # get predicted probability(predict) and class label(tag)
            uncertainty.append(predict.data.cpu().numpy()) # append probabilities to list for uncertainty sorting
            labels.append(tag.data.cpu().numpy()) # append class labels to list for matching uncertainty sorting

        uncertainty = np.array(uncertainty) # uncertainty list to array
        indices = np.argmin(uncertainty, axis=0) # uncertainty sorting and obtain smallest probabilities as largest uncertainties
        uncertainty = uncertainty[[indices], np.arange(args.batch_size)].squeeze() # obtain corresponding sorted uncertainty result for each data; 100 is batch size.

        labels = np.array(labels) # class label list to array
        labels = labels[[indices], np.arange(args.batch_size)].squeeze() # match labels to uncertainty sorted results above


        best_w = indices/(len(w)-1)*(w[-1]-w[0]) + w[0] # find the correponding weights; here, indices = i in the for loop, so it could be used for correct weight recovery

        mix_images_weights.append(best_w) # element-wise multiplication for image
        lab.append(labels) # get corresponding class labels
        unc_tags.append(uncertainty) # get corresponding uncertainty (probabilites)


    mix_images_weights = np.concatenate(mix_images_weights, axis=0)
    lab = np.concatenate(lab, axis=0)
    unc_tags = np.concatenate(unc_tags, axis=0)

    x_oris_1_array = np.concatenate(x_oris_1_array, axis=0)
    x_oris_2_array = np.concatenate(x_oris_2_array, axis=0)


    mix_img_weights = []

    x_unlabeled_oris_1 = []
    x_unlabeled_oris_2 = []

    labeled_indices = []

################ DATA SELECTION  ######################
    local_indices = np.argsort(unc_tags)[:10000]  #### TOP 10K are selected
    mix_img_weights.append( mix_images_weights[local_indices] )
    labeled_indices.append( local_indices )
#######################################################

    labeled_indices = np.concatenate(labeled_indices, axis=0)
    mix_img_weights = np.concatenate(mix_img_weights, axis=0)


    k = 0
    for j in labeled_indices:
        mix_image.append(x_oris_1_array[j][:][:][:]*mix_img_weights[k] + x_oris_2_array[j][:][:][:]*(1-mix_img_weights[k]))
        k += 1

    for j in range(len(x_oris_1)):
        if j not in labeled_indices:
            x_unlabeled_oris_1.append(x_oris_1[j])
            x_unlabeled_oris_2.append(x_oris_2[j])

    print('==> Dump Left Unlabeled Images...')
    outpath_pkl = './images/left/cifar10_left_unlabeled_'+str(len(x_unlabeled_oris_1))+'.pkl' 
    print(outpath_pkl)
    pickle.dump({'x_1_unlabeled_path':x_unlabeled_oris_1,'x_2_unlabeled_path':x_unlabeled_oris_2},open(outpath_pkl,'wb'),-1)

    print('==> Dump Query-labeled Images...')
    outpath_pkl = './images/query/cifar10_query_new_label_'+str(len(mix_image))+'.pkl'
    print(outpath_pkl)
    pickle.dump({'mix_array':mix_image},open(outpath_pkl,'wb'),-1)

if __name__ == '__main__':
    main()



