from pixyz.distributions import Normal, ProductOfNormal
from pixyz.losses import KullbackLeibler, Parameter
from pixyz.models import Model
from torch import optim
import torch
from models.utils import unsupervised_distr_no_var


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class VAEVAEThree(object):
    def __init__(self, z_dim, optimizer_params, q, p, q_double=None, q_triple=None, extra_modules=None):
        self.sample_shape = 1000
        self.z_dim = z_dim
        if extra_modules is None:
            extra_modules = []
        self.name = 'VAEVAEThree'
        self.prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                       var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

        self.q1 = q().to(device)
        self.q2 = q().replace_var(x1='y1').to(device)
        self.q3 = q().replace_var(x1='z1').to(device)

        self.px = p().replace_var(x1='x1').to(device)
        self.py = p().replace_var(x1='y1').to(device)
        self.pz = p().replace_var(x1='z1').to(device)

        self.p = self.px * self.py * self.pz

        self.beta = Parameter("beta")
        self.is_supervised = Parameter("is_supervised")

        self.q = q_triple().to(device)

        self.q_xy = q_double().to(device)
        self.q_yz = q_double().replace_var(x1='y1', y1='z1').to(device)
        self.q_xz = q_double().replace_var(x1='z1', y1='x1').to(device)

        self.px_u = unsupervised_distr_no_var(self.px).to(device)
        self.py_u = unsupervised_distr_no_var(self.py).to(device)
        self.pz_u = unsupervised_distr_no_var(self.pz).to(device)

        self.q1_u = unsupervised_distr_no_var(self.q1).to(device)
        self.q2_u = q().replace_var(x1='y1_u').to(device)
        self.q3_u = q().replace_var(x1='z1_u').to(device)

        loss_supervised_recon = -(
                self.px.log_prob().expectation(self.q) +
                self.py.log_prob().expectation(self.q) +
                self.pz.log_prob().expectation(self.q) +

                self.px.log_prob().expectation(self.q_xy) +
                self.py.log_prob().expectation(self.q_xy) +
                self.pz.log_prob().expectation(self.q_xy) +

                self.px.log_prob().expectation(self.q_yz) +
                self.py.log_prob().expectation(self.q_yz) +
                self.pz.log_prob().expectation(self.q_yz) +

                self.px.log_prob().expectation(self.q_xz) +
                self.py.log_prob().expectation(self.q_xz) +
                self.pz.log_prob().expectation(self.q_xz)
        )

        loss_supervised_kl = self.beta*(
                KullbackLeibler(self.q, self.q_xy) +
                KullbackLeibler(self.q, self.q_yz) +
                KullbackLeibler(self.q, self.q_xz) +

                KullbackLeibler(self.q_xy, self.prior) +
                KullbackLeibler(self.q_yz, self.prior) +
                KullbackLeibler(self.q_xz, self.prior) +

                KullbackLeibler(self.q_xy, self.q1) +
                KullbackLeibler(self.q_xy, self.q2) +

                KullbackLeibler(self.q_yz, self.q2) +
                KullbackLeibler(self.q_yz, self.q3) +

                KullbackLeibler(self.q_xz, self.q1) +
                KullbackLeibler(self.q_xz, self.q3)
        )

        loss_unsupervised_recon = -(
                self.px_u.log_prob().expectation(self.q1_u) +
                self.py_u.log_prob().expectation(self.q2_u) +
                self.pz_u.log_prob().expectation(self.q3_u)
        )

        loss_unsupervised_kl = self.beta*(
                KullbackLeibler(self.q1_u, self.prior) +
                KullbackLeibler(self.q2_u, self.prior) +
                KullbackLeibler(self.q3_u, self.prior)
        )

        loss = loss_supervised_recon.mean() + loss_supervised_kl.mean() + loss_unsupervised_recon.mean() + loss_unsupervised_kl.mean()

        self.model = Model(
            loss=loss,
            distributions=extra_modules + [
                self.px,
                self.py,
                self.pz,

                self.q1,
                self.q2,
                self.q3,

                self.q,

                self.q_xy,
                self.q_yz,
                self.q_xz,
            ],
            optimizer=optim.Adam,
            optimizer_params=optimizer_params
        )

    def model_args(self, x, y, z, xu, yu, zu, beta=1.0):
        return {
            "x1": x, "y1": y, "z1": z,
            "x1_u": xu, "y1_u": yu, "z1_u": zu,
            "beta": beta,
        }

    def eval_args(self, x, y, z):
        return {
            "x1": x, "y1": y, "z1": z,
            "x1_u": x, "y1_u": y, "z1_u": z,
            "beta": 1.0,
        }

    def sample_z_from_x(self, x, sample_shape=None):
        if sample_shape:
            return self.q1.sample({"x1": x}, sample_shape=[sample_shape], return_all=False)
        return self.q1.sample({"x1": x}, return_all=False)

    def sample_z_from_y(self, y, sample_shape=None):
        if sample_shape:
            return self.q2.sample({"y1": y}, sample_shape=[sample_shape], return_all=False)
        return self.q2.sample({"y1": y}, return_all=False)

    def sample_z_from_z(self, z, sample_shape=None):
        if sample_shape:
            return self.q3.sample({"z1": z}, sample_shape=[sample_shape], return_all=False)
        return self.q3.sample({"z1": z}, return_all=False)

    def sample_z_all(self, x, y, z, sample_shape=None):
        if sample_shape:
            return self.q.sample({"x1": x, "y1": y, "z1": z}, sample_shape=[sample_shape], return_all=False)
        return self.q.sample({"x1": x, "y1": y, "z1": z}, return_all=False)

    def reconstruct_x(self, z):
        return self.px.sample_mean(z)

    def reconstruct_y(self, z):
        return self.py.sample_mean(z)

    def reconstruct_z(self, z):
        return self.pz.sample_mean(z)

    def sample_prior(self, sample_shape=None):
        if sample_shape:
            return self.prior.sample(sample_shape=[sample_shape], return_all=False)
        return self.prior.sample(return_all=False)

    def get_number_of_parameters(self):
        result = 0
        for m in [
                self.px,
                self.py,
                self.pz,

                self.q1,
                self.q2,
                self.q3,

                self.q,

                self.q_xy,
                self.q_yz,
                self.q_xz,
        ]:
            result += sum(p.numel() for p in m.parameters() if p.requires_grad)
        return result
