# MNIST model specification

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod, sqrt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from utils import Constants
from vis import plot_embeddings, plot_kls_df
from .vae import VAE

# Constants
dataSize = torch.Size([1, 14, 28])
data_dim = int(prod(dataSize))
hidden_dim = 400


def extra_hidden_layer():
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))


# Classes
class Enc(nn.Module):
    """ Generate latent parameters for MNIST image data. """

    def __init__(self, latent_dim, num_hidden_layers=1):
        super(Enc, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
        )
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        mean, var = self.fc21(x), self.fc22(x)
        return mean, F.softplus(var)


class Dec(nn.Module):
    """ Generate an MNIST image given a sample from the latent space. """

    def __init__(self, latent_dim, num_hidden_layers=1):
        super(Dec, self).__init__()

        self.n_latents = latent_dim
        self.upsampler = nn.Sequential(
            nn.Linear(self.n_latents, 512),
            nn.ReLU(),
            nn.Linear(512, 112 * 1 * 1),
            nn.ReLU())
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(112, 56, kernel_size=(3, 5), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(56, 28, kernel_size=(3, 5), stride=2),
            nn.ReLU(),
        )
        self.upsample = nn.ConvTranspose2d(28, 1, kernel_size=(2, 4), stride=2)

    def forward(self, z):
        z = self.upsampler(z)
        k, batch, dim = z.size()
        z = z.view(k*batch, dim, 1, 1)
        z = self.hallucinate(z)
        z = self.upsample(z, output_size=[14, 28])
        z = torch.sigmoid(z.view(k, batch, 1, 14, 28))
        return z, torch.tensor(0.75).to(z.device)  # mean, length scale


def crop_half(img, half):
    N = 28
    # print('IMG.size', img.size)
    up = transforms.functional.crop(img, 0, 0, N/2, N)
    # print('UP.size', up.size)
    down = transforms.functional.crop(img, N/2, 0, N/2, N)
    if half == 'top':
        return up
    elif half == 'down':
        return down


class MNIST_half(VAE):
    """ Derive a specific sub-class of a VAE for MNIST. """

    def __init__(self, params):
        super(MNIST_half, self).__init__(
            dist.Laplace,  # prior
            dist.Laplace,  # likelihood
            dist.Laplace,  # posterior
            Enc(params.latent_dim, params.num_hidden_layers),
            Dec(params.latent_dim, params.num_hidden_layers),
            params
        )
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'mnist_split'
        self.dataSize = dataSize
        self.llik_scaling = 1.

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    @staticmethod
    def getDataLoaders(batch_size, half, sampler_train, sampler_test, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        tx = transforms.Compose([
            transforms.Lambda(lambd=lambda x: transforms.ToTensor()(crop_half(x, half)))
        ])
        train = DataLoader(datasets.MNIST('../data', train=True, download=True, transform=tx),
                           batch_size=batch_size, sampler=sampler_train, **kwargs)
        test = DataLoader(datasets.MNIST('../data', train=False, download=True, transform=tx),
                          batch_size=batch_size, sampler=sampler_test, **kwargs)
        return train, test

    def generate(self, runPath, epoch):
        N, K = 64, 9
        samples = super(MNIST_half, self).generate(N, K).cpu()
        # wrangle things so they come out tiled
        samples = samples.view(K, N, *samples.size()[1:]).transpose(0, 1)  # N x K x 1 x 28 x 28
        s = [make_grid(t, nrow=int(sqrt(K)), padding=0) for t in samples]
        save_image(torch.stack(s),
                   '{}/gen_samples_{:03d}.png'.format(runPath, epoch),
                   nrow=int(sqrt(N)))

    def reconstruct(self, data, runPath, epoch):
        recon = super(MNIST_half, self).reconstruct(data[:8])
        comp = torch.cat([data[:8], recon]).data.cpu()
        save_image(comp, '{}/recon_{:03d}.png'.format(runPath, epoch))

    def analyse(self, data, runPath, epoch):
        zemb, zsl, kls_df = super(MNIST_half, self).analyse(data, K=10)
        labels = ['Prior', self.modelName.lower()]
        plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch))
        plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch))
