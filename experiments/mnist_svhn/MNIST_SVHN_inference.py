from pixyz.distributions import Normal, Bernoulli

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from numpy import prod


# Constants
dataSize = torch.Size([1, 28, 28])
data_dim = int(prod(dataSize))

dataSize_svhn = torch.Size([3, 32, 32])
imgChans = dataSize_svhn[0]
fBase = 32  # base size of filter channels for SVHN

hidden_dim = 400
num_hidden_layers = 1
latent_dim = 20
z_dim = latent_dim


def extra_hidden_layer():
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))


class InferenceJoint(Normal):
    def __init__(self, n_inner=512, coef=1):

        super(InferenceJoint, self).__init__(cond_var=["x1", "y1"], var=["z"], name="q_joint")

        modules = []
        modules.append(nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer() for _ in range(num_hidden_layers - 1)])
        self.features_x = nn.Sequential(*modules)

        self.features_y = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(imgChans, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
            nn.Conv2d(fBase * 4, hidden_dim, 4, 1, 0, bias=True),
        )

        self.fc21 = nn.Linear(2*hidden_dim, latent_dim)
        self.fc22 = nn.Linear(2*hidden_dim, latent_dim)

    def forward(self, x1, y1):
        x1 = x1.view(*x1.size()[:-3], -1)
        x = self.features_x(x1)
        y = self.features_y(y1).squeeze()
        e = torch.cat([x, y], 1)
        return {"loc": self.fc21(e), "scale": F.softplus(self.fc22(e))}


# inference model q1(z|x)
class InferenceX(Normal):
    def __init__(self, n_inner=512, coef=1):

        super(InferenceX, self).__init__(cond_var=["x1"], var=["z"], name="q1")

        modules = []
        modules.append(nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer() for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x1):
        e = self.enc(x1.view(*x1.size()[:-3], -1))  # flatten data
        return {"loc": self.fc21(e), "scale": F.softplus(self.fc22(e))}


# inference model q*2(z|x)
class InferenceX_missing(Normal):
    def __init__(self, n_inner=512, coef=1):
        super(InferenceX_missing, self).__init__(cond_var=["x2"], var=["z"], name="q1")

        modules = []
        modules.append(nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer() for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x2):
        e = self.enc(x2.view(*x2.size()[:-3], -1))
        return {"loc": self.fc21(e), "scale": F.softplus(self.fc22(e))}


# inference model q2(z|y)
class InferenceY(Normal):
    def __init__(self, n_inner=512, coef=1):

        super(InferenceY, self).__init__(cond_var=["y1"], var=["z"], name="q2")

        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(imgChans, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
        )
        self.c1 = nn.Conv2d(fBase * 4, latent_dim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(fBase * 4, latent_dim, 4, 1, 0, bias=True)

    def forward(self, y1):
        e = self.enc(y1)
        return {"loc": self.c1(e).squeeze(), "scale": F.softplus(self.c2(e).squeeze())}


# inference model q*1(z|y)
class InferenceY_missing(Normal):
    def __init__(self, n_inner=512, coef=1):

        super(InferenceY_missing, self).__init__(cond_var=["y2"], var=["z"], name="q_star_1")

        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(imgChans, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
        )
        self.c1 = nn.Conv2d(fBase * 4, latent_dim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(fBase * 4, latent_dim, 4, 1, 0, bias=True)

    def forward(self, y2):
        e = self.enc(y2)
        return {"loc": self.c1(e).squeeze(), "scale": F.softplus(self.c2(e).squeeze())}


# generative model p(x|z)
class GeneratorX(Bernoulli):
    def __init__(self):
        super(GeneratorX, self).__init__(cond_var=["z"], var=["x1"], name="p_x")

        modules = []
        modules.append(nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer() for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(hidden_dim, data_dim)

    def forward(self, z):
        p = self.fc3(self.dec(z))
        return {"probs": torch.sigmoid(p.view(*z.size()[:-1], *dataSize))}


# generative model p(x|z)
class GeneratorY(Bernoulli):
    def __init__(self):
        super(GeneratorY, self).__init__(cond_var=["z"], var=["y1"], name="p_y")

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, fBase * 4, 4, 1, 0, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.ConvTranspose2d(fBase, imgChans, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # Output size: 3 x 32 x 32
        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        out = out.view(*z.size()[:-3], *out.size()[1:])
        return {"probs": out}


class SVHN_Classifier(nn.Module):
    def __init__(self):
        super(SVHN_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class MNIST_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
