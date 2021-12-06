#!/usr/bin/env python
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torchvision
import torchvision.transforms as transforms

from models.vgg import VGG
import pickle
import numpy as np
from my_loader import ImageFolder

# arguments
parser = argparse.ArgumentParser(description='Active Mixup')
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--root', type=str, required=True)
parser.add_argument('--real_images', default=500, type=int, metavar='N')
parser.add_argument('--batch_size', default=100, type=int, metavar='N')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

opt = parser.parse_args()



# dataset
print('==> Preparing data..')
if opt.data == 'cifar10':
    nclass = 10
    img_width = 32
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainloader = torch.utils.data.DataLoader(
        ImageFolder(transform=transform_test, root=opt.root),
        batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.workers, pin_memory=True)
else:
    raise ValueError(f'invlid dataset: {opt.data}')

cudnn.benchmark = True

x_oris = []
x_oris_img = []

for it, (img, _, img_path) in enumerate(trainloader):
    if it < opt.real_images/opt.batch_size: 
        x_oris.extend(img_path)
        x_oris_img.append(img)
    else:
        break


################# Dump Real Image Array #################

print('==> Dump real image data..')
x_oris_img = np.concatenate(x_oris_img, axis=0)
outpath_pkl = './images/cifar10_real_images_'+str(len(x_oris))+'.pkl'
print(outpath_pkl)
pickle.dump({'x_oris':x_oris_img},open(outpath_pkl,'wb'),-1)



################# Dump Mixed Image Array #################

print('==> Dump mixed image data..')

### Mix Image by Index Rotation ###
x_comb_index = []
x_seq_index = np.arange(opt.real_images)
for i in range(int((opt.real_images-1)/2)):
    x_comb_index.append([x_seq_index, np.roll(x_seq_index, i+1)])


### Retrieve Indeices and Dump Image Array ###
x_comb_index = np.array(x_comb_index)

x_oris_1 = []
x_oris_2 = []

for j in range( x_comb_index.shape[2]):
    if(j%100==0):
        print('Appended Mixed Images: ', j+100)
    for i in range(x_comb_index.shape[0]):
        x_oris_1.append(x_oris[x_comb_index[i,0,j]])
        x_oris_2.append(x_oris[x_comb_index[i,1,j]])

outpath_pkl = './images/cifar10_mix_images_'+str(len(x_oris_1))+'.pkl'
print(outpath_pkl)
pickle.dump({'x_1_unlabeled_path':x_oris_1, 'x_2_unlabeled_path':x_oris_2},open(outpath_pkl,'wb'),-1)    
