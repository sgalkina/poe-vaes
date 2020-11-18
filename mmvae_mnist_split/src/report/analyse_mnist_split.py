"""Calculate cross and joint coherence of trained model on MNIST-Split dataset.
Train and evaluate a linear model for latent space digit classification."""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# relative import hacks (sorry)
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) # for bash user
os.chdir(parentdir) # for pycharm user

import models
from helper import Latent_Classifier, SVHN_Classifier, MNIST_Classifier
from utils import Logger, Timer


torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Analysing MM-DGM results')
parser.add_argument('--save-dir', type=str, default="",
                    metavar='N', help='save directory of results')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA use')
cmds = parser.parse_args()
runPath = cmds.save_dir

print('runPath', runPath, flush=True)

# sys.stdout = Logger('{}/ms_acc.log'.format(runPath))
args = torch.load(runPath + '/args.rar')


print("BEGIN ANALYSE", flush=True)

# cuda stuff
needs_conversion = cmds.no_cuda and args.cuda
conversion_kwargs = {'map_location': lambda st, loc: st} if needs_conversion else {}
args.cuda = not cmds.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
torch.manual_seed(args.seed)

print("BEFORE MODEL", flush=True)

modelC = getattr(models, 'VAE_{}'.format(args.model))
model = modelC(args)
if args.cuda:
    model.cuda()

model.load_state_dict(torch.load(runPath + '/model.rar', **conversion_kwargs), strict=False)
B = 256  # rough batch size heuristic
train_loader, test_loader = model.getDataLoaders(B, device=device)
N = len(test_loader.dataset)


def coherence(epochs):
    model.eval()

    mnist_net = MNIST_Classifier().to(device)
    mnist_net.load_state_dict(torch.load('../data/mnist_model.pt'))
    mnist_net.eval()

    total = 0
    corr_m = 0
    corr_s = 0
    corr_b = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            mnist_top, mnist_bottom, targets = unpack_data_mlp(data, option='both')
            mnist_top, mnist_bottom, targets = mnist_top.to(device), mnist_bottom.to(device), targets.to(device)
            _, px_zs, _ = model([mnist_top, mnist_bottom], 1)

            from_up = torch.cat([px_zs[1][0].mean.squeeze(0), px_zs[1][1].mean.squeeze(0)], 2)
            from_down = torch.cat([px_zs[0][0].mean.squeeze(0), px_zs[0][1].mean.squeeze(0)], 2)
            from_both = torch.cat([px_zs[0][0].mean.squeeze(0), px_zs[1][1].mean.squeeze(0)], 2)

            mnist_mnist = mnist_net(from_up)
            svhn_svhn = mnist_net(from_down)
            both = mnist_net(from_both)

            _, pred_m = torch.max(mnist_mnist.data, 1)
            _, pred_s = torch.max(svhn_svhn.data, 1)
            _, pred_b = torch.max(both.data, 1)
            total += targets.size(0)
            corr_m += (pred_m == targets).sum().item()
            corr_s += (pred_s == targets).sum().item()
            corr_b += (pred_b == targets).sum().item()

    print('Cross coherence: \n from top {:.2f}% \n from bottom {:.2f}%'.format(
        corr_m / total * 100, corr_s / total * 100))
    print('Joint coherence\n {:.2f}%'.format(
        corr_b / total * 100))


def unpack_data_mlp(dataB, option='both'):
    if len(dataB[0]) == 2:
        if option == 'both':
            return dataB[0][0], dataB[1][0], dataB[1][1]
        elif option == 'svhn':
            return dataB[1][0], dataB[1][1]
        elif option == 'mnist':
            return dataB[0][0], dataB[0][1]
    else:
        return dataB


def unpack_model(option='svhn'):
    if 'mnist_svhn' in args.model:
        return model.vaes[1] if option == 'svhn' else model.vaes[0]
    else:
        return model


print('\n' + '-' * 45 + ' coherence' + '-' * 45, flush=True)
coherence(epochs=30)
#

