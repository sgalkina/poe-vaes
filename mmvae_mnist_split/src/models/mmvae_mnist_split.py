# MNIST-SVHN multi-modal model specification
import os

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import sqrt, prod
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset
from torch.utils.data.sampler import SequentialSampler
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

from vis import plot_embeddings, plot_kls_df
from .mmvae import MMVAE
from .vae_mnist_split import MNIST_half


class MNIST_Split(MMVAE):
    def __init__(self, params):
        super(MNIST_Split, self).__init__(dist.Laplace, params, MNIST_half, MNIST_half)
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.vaes[0].llik_scaling = prod(self.vaes[1].dataSize) / prod(self.vaes[0].dataSize) \
            if params.llik_scaling == 0 else params.llik_scaling
        self.modelName = 'mnist-svhn'

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):

        def get_sampler(N_data):
            indices = list(range(N_data))
            np.random.shuffle(indices)
            return SequentialSampler(indices)

        sampler_train = get_sampler(60000)
        sampler_test = get_sampler(10000)

        # load base datasets
        t1, s1 = self.vaes[0].getDataLoaders(batch_size, 'top', sampler_train, sampler_test, device)
        t2, s2 = self.vaes[1].getDataLoaders(batch_size, 'down', sampler_train, sampler_test, device)

        train_mnist_svhn = TensorDataset([
            t1.dataset, t2.dataset
        ])
        test_mnist_svhn = TensorDataset([
            s1.dataset, s2.dataset
        ])

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_mnist_svhn, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_mnist_svhn, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train, test

    def generate(self, runPath, epoch):
        N = 64
        samples_list = super(MNIST_Split, self).generate(N)
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            print('samples.shape', samples.shape)
            # wrangle things so they come out tiled
            samples = samples.view(N, *samples.size()[1:])
            print('samples.shape after view', samples.shape)
            save_image(samples,
                       '{}/gen_samples_{}_{:03d}.png'.format(runPath, i, epoch),
                       nrow=int(sqrt(N)))

    def reconstruct(self, data, runPath, epoch):
        recons_mat = super(MNIST_Split, self).reconstruct([d[:8] for d in data])
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                _data = data[r][:8].cpu()
                recon = recon.squeeze(0).cpu()
                plt.imshow(recon[0][0].cpu().numpy())
                plt.savefig(f'recon_{epoch}.png')
                # resize mnist to 32 and colour. 0 => mnist, 1 => svhn
                # _data = _data if r == 1 else resize_img(_data, self.vaes[1].dataSize)
                # recon = recon if o == 1 else resize_img(recon, self.vaes[1].dataSize)
                comp = torch.cat([_data, recon])
                save_image(comp, '{}/recon_{}x{}_{:03d}.png'.format(runPath, r, o, epoch))

    def analyse(self, data, runPath, epoch):
        zemb, zsl, kls_df = super(MNIST_Split, self).analyse(data, K=10)
        labels = ['Prior', *[vae.modelName.lower() for vae in self.vaes]]
        plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch))
        plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch))

