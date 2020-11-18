from pixyz.distributions import Normal, Bernoulli

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import ResNet, BasicBlock

y_dim = 10
z_dim = 64


class InferenceJointAll(Normal):
    def __init__(self, n_inner=512, coef=1):

        super(InferenceJointAll, self).__init__(cond_var=["x1", "y1", "z1"], var=["z"], name="q_joint")

        self.features_x = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.ReLU(),
        )
        self.features_y = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.ReLU(),
        )
        self.features_z = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(768*3, int(n_inner * coef)),
            nn.ReLU(),
            nn.Linear(int(n_inner * coef), z_dim * 2))
        self.n_latents = z_dim

    def forward(self, x1, y1, z1):
        n_latents = self.n_latents
        x = self.features_x(x1)
        x = x.view(x.size(0), -1)
        y = self.features_y(y1)
        y = y.view(y.size(0), -1)
        z = self.features_z(z1)
        z = z.view(y.size(0), -1)
        x = self.classifier(torch.cat([x, y, z], 1))
        return {"loc": x[:, :n_latents], "scale": F.softplus(x[:, n_latents:])}


class InferenceJoint(Normal):
    def __init__(self, n_inner=512, coef=1):

        super(InferenceJoint, self).__init__(cond_var=["x1", "y1"], var=["z"], name="q_joint")

        self.features_x = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.ReLU(),
        )
        self.features_y = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(768*2, int(n_inner * coef)),
            nn.ReLU(),
            nn.Linear(int(n_inner * coef), z_dim * 2))
        self.n_latents = z_dim

    def forward(self, x1, y1):
        n_latents = self.n_latents
        x = self.features_x(x1)
        x = x.view(x.size(0), -1)
        y = self.features_y(y1)
        y = y.view(y.size(0), -1)
        x = self.classifier(torch.cat([x, y], 1))
        return {"loc": x[:, :n_latents], "scale": F.softplus(x[:, n_latents:])}


# inference model q1(z|x)
class InferenceX(Normal):
    def __init__(self, n_inner=512, coef=1):

        super(InferenceX, self).__init__(cond_var=["x1"], var=["z"], name="q1")

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, int(n_inner * coef)),
            nn.ReLU(),
            nn.Linear(int(n_inner * coef), z_dim * 2))
        self.n_latents = z_dim

    def forward(self, x1):
        n_latents = self.n_latents
        x = self.features(x1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return {"loc": x[:, :n_latents], "scale": F.softplus(x[:, n_latents:])}


# generative model p(x|z)
class GeneratorX(Bernoulli):
    def __init__(self):
        super(GeneratorX, self).__init__(cond_var=["z"], var=["x1"], name="p_x")

        self.n_latents = z_dim
        self.upsampler = nn.Sequential(
            nn.Linear(z_dim, 512),
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
        z = z.view(z.size(0), 112, 1, 1)
        z = self.hallucinate(z)
        z = self.upsample(z, output_size=[14, 28])
        return {"probs": torch.sigmoid(z)}


class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3), bias=False)
