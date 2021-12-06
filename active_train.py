#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torchvision
import torchvision.transforms as transforms

import sys, os
import argparse

import numpy as np

#from my_loader import active_learning_loader
from my_loader import active_learning_loader

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', required=True, type=float, help='learning rate')
parser.add_argument('--data', required=True, type=str, help='dataset name')
parser.add_argument('--model', required=True, type=str, help='model name')
parser.add_argument('--root', required=True, type=str, help='path to dataset')
parser.add_argument('--model_out', required=True, type=str, help='output path')
parser.add_argument('--resume', action='store_true', help='Resume training')
opt = parser.parse_args()




cuda = torch.cuda.is_available()
torch.backends.cudnn.benchmark = False


# Data
print('==> Preparing data..')
if opt.data == 'cifar10':
    nclass = 10
    img_width = 32
   
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
   
    train_dataset = active_learning_loader(
    transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]))
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True,
        num_workers=2, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root=opt.root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

else:
    raise NotImplementedError('Invalid dataset')

# Model
if opt.model == 'vgg':
    from models.vgg import VGG
    net = nn.DataParallel(VGG('VGG16', nclass, img_width=img_width).cuda())
elif opt.model == 'resnet':
    from models.resnet import ResNet34
    net = nn.DataParallel(ResNet34().cuda())
else:
    raise NotImplementedError('Invalid model')

#checkpoint = torch.load('./checkpoint/cifar10_vgg16_teacher.pth')
#net.load_state_dict(checkpoint)

# Loss function
criterion = nn.CrossEntropyLoss()

def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

# Training
def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, logits, targets) in enumerate(trainloader):
        inputs, logits, targets = inputs.cuda(), logits.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = cross_entropy(outputs*1, logits*1)
        loss.backward()
        optimizer.step()
        pred = torch.max(outputs, dim=1)[1]
        correct += torch.sum(pred.eq(targets)).item()
        total += targets.numel()
    print(f'[TRAIN] Acc: {100.*correct/total:.3f}')

# global variable
best = 0

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f'[TEST] Acc: {100.*correct/total:.3f}')

    global best

    if epoch > 80 and  100.*correct/total > best:
        best = 100.*correct/total
        torch.save(net.state_dict(), opt.model_out)
        print(f'[SAVED BEST MODEL HERE] Acc: {100.*correct/total:.3f}')

if opt.data == 'cifar10':
    epochs = [80, 60, 40, 20]

count = 0


for epoch in epochs:
    optimizer = SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5.0e-4)
    for _ in range(epoch):
        train(count)
        test(count)
        count += 1
    opt.lr /= 10

