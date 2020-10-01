from pixyz.distributions import Normal, Bernoulli, Categorical, ProductOfNormal, Laplace

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from pixyz.utils import get_dict_values, sum_samples
from .helper import FakeCategorical


# Constants
imgChans = 3
fBase_y = 64
maxSentLen = 32  # max length of any description for birds dataset
minOccur = 3
embeddingDim = 128
lenWindow = 3
fBase_x = 32
vocabSize = 1590
latentDim = 64
latent_dim, n_c = 64, 2048


class InferenceJoint(Normal):
    def __init__(self, n_inner=512, coef=1):

        super(InferenceJoint, self).__init__(cond_var=["x1", "y1"], var=["z"], name="q_joint")
        self.embedding = nn.Embedding(vocabSize, embeddingDim, padding_idx=0)
        self.features_x = nn.Sequential(
            # input size: 1 x 32 x 128
            nn.Conv2d(1, fBase_x, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase_x),
            nn.ReLU(True),
            # size: (fBase) x 16 x 64
            nn.Conv2d(fBase_x, fBase_x * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase_x * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 32
            nn.Conv2d(fBase_x * 2, fBase_x * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase_x * 4),
            nn.ReLU(True),
            # # size: (fBase * 4) x 4 x 16
            nn.Conv2d(fBase_x * 4, fBase_x * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase_x * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.Conv2d(fBase_x * 4, fBase_x * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase_x * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
        )

        dim_hidden = 256
        self.features_y = nn.Sequential()
        for i in range(int(torch.tensor(n_c / dim_hidden).log2())):
            self.features_y.add_module("layer" + str(i), nn.Sequential(
                nn.Linear(n_c // (2 ** i), n_c // (2 ** (i + 1))),
                nn.ELU(inplace=True),
            ))

        self.classifier = nn.Sequential(
            nn.Linear(2304, int(n_inner * coef)),
            nn.ReLU(),
            nn.Linear(int(n_inner * coef), latentDim * 2))

        self.n_latents = latentDim

    def forward(self, x1, y1):
        x1 = x1.argmax(axis=2)
        x = self.features_x(self.embedding(x1.long()).unsqueeze(1))
        x = x.view(x.size(0), -1)
        y = self.features_y(y1)
        # print('joint shapes', x.shape, y.shape)
        x = self.classifier(torch.cat([x, y], 1))
        return {"loc": x[:, :self.n_latents], "scale": F.softplus(x[:, self.n_latents:])}


# inference model q1(z|x)
class InferenceX(Normal):
    def __init__(self, n_inner=512, coef=1):

        super(InferenceX, self).__init__(cond_var=["x1"], var=["z"], name="q1")
        self.embedding = nn.Embedding(vocabSize, embeddingDim, padding_idx=0)
        self.enc = nn.Sequential(
            # input size: 1 x 32 x 128
            nn.Conv2d(1, fBase_x, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase_x),
            nn.ReLU(True),
            # size: (fBase) x 16 x 64
            nn.Conv2d(fBase_x, fBase_x * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase_x * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 32
            nn.Conv2d(fBase_x * 2, fBase_x * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase_x * 4),
            nn.ReLU(True),
            # # size: (fBase * 4) x 4 x 16
            nn.Conv2d(fBase_x * 4, fBase_x * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase_x * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.Conv2d(fBase_x * 4, fBase_x * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase_x * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
        )
        self.c1 = nn.Conv2d(fBase_x * 4, latentDim, 4, 1, 0, bias=False)
        self.c2 = nn.Conv2d(fBase_x * 4, latentDim, 4, 1, 0, bias=False)
        # c1, c2 size: latentDim x 1 x 1

    def forward(self, x1):
        x1 = x1.argmax(axis=2)
        e = self.enc(self.embedding(x1.long()).unsqueeze(1))
        mu, logvar = self.c1(e).squeeze(), self.c2(e).squeeze()
        return {"loc": mu, "scale": F.softplus(logvar)}


# inference model q*2(z|x)
class InferenceX_missing(Normal):
    def __init__(self, n_inner=512, coef=1):

        super(InferenceX_missing, self).__init__(cond_var=["x2"], var=["z"], name="q_star_2")
        self.embedding = nn.Embedding(vocabSize, embeddingDim, padding_idx=0)
        self.enc = nn.Sequential(
            # input size: 1 x 32 x 128
            nn.Conv2d(1, fBase_x, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase_x),
            nn.ReLU(True),
            # size: (fBase) x 16 x 64
            nn.Conv2d(fBase_x, fBase_x * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase_x * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 32
            nn.Conv2d(fBase_x * 2, fBase_x * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase_x * 4),
            nn.ReLU(True),
            # # size: (fBase * 4) x 4 x 16
            nn.Conv2d(fBase_x * 4, fBase_x * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase_x * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.Conv2d(fBase_x * 4, fBase_x * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase_x * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
        )
        self.c1 = nn.Conv2d(fBase_x * 4, latentDim, 4, 1, 0, bias=False)
        self.c2 = nn.Conv2d(fBase_x * 4, latentDim, 4, 1, 0, bias=False)
        # c1, c2 size: latentDim x 1 x 1

    def forward(self, x2):
        x2 = x2.argmax(axis=2)
        e = self.enc(self.embedding(x2.long()).unsqueeze(1))
        mu, logvar = self.c1(e).squeeze(), self.c2(e).squeeze()
        return {"loc": mu, "scale": F.softplus(logvar)}


# inference model q2(z|y)
class InferenceY(Normal):
    def __init__(self, n_inner=512, coef=1):

        super(InferenceY, self).__init__(cond_var=["y1"], var=["z"], name="q2")
        dim_hidden = 256
        self.enc = nn.Sequential()
        for i in range(int(torch.tensor(n_c / dim_hidden).log2())):
            self.enc.add_module("layer" + str(i), nn.Sequential(
                nn.Linear(n_c // (2 ** i), n_c // (2 ** (i + 1))),
                nn.ELU(inplace=True),
            ))
        # relies on above terminating at dim_hidden
        self.fc21 = nn.Linear(dim_hidden, latent_dim)
        self.fc22 = nn.Linear(dim_hidden, latent_dim)

    def forward(self, y1):
        e = self.enc(y1)
        return {"loc": self.fc21(e), "scale": F.softplus(self.fc22(e))}


# inference model q*1(z|y)
class InferenceY_missing(Normal):
    def __init__(self, n_inner=512, coef=1):

        super(InferenceY_missing, self).__init__(cond_var=["y2"], var=["z"], name="q_star_1")
        dim_hidden = 256
        self.enc = nn.Sequential()
        for i in range(int(torch.tensor(n_c / dim_hidden).log2())):
            self.enc.add_module("layer" + str(i), nn.Sequential(
                nn.Linear(n_c // (2 ** i), n_c // (2 ** (i + 1))),
                nn.ELU(inplace=True),
            ))
        # relies on above terminating at dim_hidden
        self.fc21 = nn.Linear(dim_hidden, latent_dim)
        self.fc22 = nn.Linear(dim_hidden, latent_dim)
        # c1, c2 size: latentDim x 1 x 1

    def forward(self, y2):
        e = self.enc(y2)
        return {"loc": self.fc21(e), "scale": F.softplus(self.fc22(e))}


# inference model q2(z|y)
class InferenceYRaw(Normal):
    def __init__(self, n_inner=512, coef=1):

        super(InferenceYRaw, self).__init__(cond_var=["y1"], var=["z"], name="q2")
        modules = [
            # input size: 3 x 128 x 128
            nn.Conv2d(imgChans, fBase_y, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # input size: 1 x 64 x 64
            nn.Conv2d(fBase_y, fBase_y * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 32 x 32
            nn.Conv2d(fBase_y * 2, fBase_y * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 16 x 16
            nn.Conv2d(fBase_y * 4, fBase_y * 8, 4, 2, 1, bias=True),
            nn.ReLU(True)]
        # size: (fBase * 8) x 4 x 4

        self.enc = nn.Sequential(*modules)
        self.c1 = nn.Conv2d(fBase_y * 8, latentDim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(fBase_y * 8, latentDim, 4, 1, 0, bias=True)
        # c1, c2 size: latentDim x 1 x 1

    def forward(self, y1):
        e = self.enc(y1)
        return {"loc": self.c1(e).squeeze(), "scale": F.softplus(self.c2(e)).squeeze()}


# inference model q*1(z|y)
class InferenceY_missingRaw(Normal):
    def __init__(self, n_inner=512, coef=1):

        super(InferenceY_missingRaw, self).__init__(cond_var=["y2"], var=["z"], name="q_star_1")
        modules = [
            # input size: 3 x 128 x 128
            nn.Conv2d(imgChans, fBase_y, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # input size: 1 x 64 x 64
            nn.Conv2d(fBase_y, fBase_y * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 32 x 32
            nn.Conv2d(fBase_y * 2, fBase_y * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 16 x 16
            nn.Conv2d(fBase_y * 4, fBase_y * 8, 4, 2, 1, bias=True),
            nn.ReLU(True)]
        # size: (fBase * 8) x 4 x 4

        self.enc = nn.Sequential(*modules)
        self.c1 = nn.Conv2d(fBase_y * 8, latentDim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(fBase_y * 8, latentDim, 4, 1, 0, bias=True)
        # c1, c2 size: latentDim x 1 x 1

    def forward(self, y2):
        e = self.enc(y2)
        return {"loc": self.c1(e).squeeze(), "scale": F.softplus(self.c2(e)).squeeze()}


# generative model p(x|z)
class GeneratorX(FakeCategorical):
    def __init__(self):
        super(GeneratorX, self).__init__(cond_var=["z"], var=["x1"], name="p_x")
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latentDim, fBase_x * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fBase_x * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
            nn.ConvTranspose2d(fBase_x * 4, fBase_x * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase_x * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.ConvTranspose2d(fBase_x * 4, fBase_x * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase_x * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 32
            nn.ConvTranspose2d(fBase_x * 4, fBase_x * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase_x * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 64
            nn.ConvTranspose2d(fBase_x * 2, fBase_x, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase_x),
            nn.ReLU(True),
            # size: (fBase) x 32 x 128
            nn.ConvTranspose2d(fBase_x, 1, 4, 2, 1, bias=False),
            nn.ReLU(True)
            # Output size: 1 x 64 x 256
        )
        # inverts the 'embedding' module upto one-hotness
        self.toVocabSize = nn.Linear(embeddingDim, vocabSize)

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:])).view(-1, embeddingDim)
        out = self.toVocabSize(out).view(*z.size()[:-3], maxSentLen, vocabSize)
        # out = F.softmax(out, dim=1)
        return {"probs": out}


# generative model p(y|z)
class GeneratorYRaw(Bernoulli):
    def __init__(self):
        super(GeneratorYRaw, self).__init__(cond_var=["z"], var=["y1"], name="p_y")
        modules = [nn.ConvTranspose2d(latentDim, fBase_y * 8, 4, 1, 0, bias=True),
                   nn.ReLU(True), ]

        modules.extend([
            nn.ConvTranspose2d(fBase_y * 8, fBase_y * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 16 x 16
            nn.ConvTranspose2d(fBase_y * 4, fBase_y * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 32 x 32
            nn.ConvTranspose2d(fBase_y * 2, fBase_y, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 64 x 64
            nn.ConvTranspose2d(fBase_y, imgChans, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # Output size: 3 x 128 x 128
        ])
        self.dec = nn.Sequential(*modules)

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        out = out.view(*z.size()[:-3], *out.size()[1:])
        return {"probs": torch.sigmoid(out)}


# generative model p(y|z)
class GeneratorY(Laplace):
    def __init__(self):
        super(GeneratorY, self).__init__(cond_var=["z"], var=["y1"], name="p_y")
        self.n_c = n_c
        dim_hidden = 256
        self.dec = nn.Sequential()
        for i in range(int(torch.tensor(n_c / dim_hidden).log2())):
            indim = latent_dim if i == 0 else dim_hidden * i
            outdim = dim_hidden if i == 0 else dim_hidden * (2 * i)
            self.dec.add_module("out_t" if i == 0 else "layer" + str(i) + "_t", nn.Sequential(
                nn.Linear(indim, outdim),
                nn.ELU(inplace=True),
            ))
        # relies on above terminating at n_c // 2
        self.fc31 = nn.Linear(n_c // 2, n_c)

    def forward(self, z):
        p = self.dec(z.view(-1, z.size(-1)))
        mean = self.fc31(p).view(*z.size()[:-1], -1)
        return {"loc": mean, "scale": torch.tensor([0.01]).to(mean.device)}
