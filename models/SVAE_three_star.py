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


class SVAEThreeStar(object):
    def __init__(self, z_dim, optimizer_params, q, p, q_double=None, q_triple=None, extra_modules=None):
        self.sample_shape = 1000
        self.z_dim = z_dim
        if extra_modules is None:
            extra_modules = []
        self.name = 'SVAEThreeStar'
        self.prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                       var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

        self.q11 = q().replace_var(x1='x1').to(device)
        self.q12 = q().replace_var(x1='x2').to(device)
        self.q13 = q().replace_var(x1='x3').to(device)

        self.q21 = q().replace_var(x1='y1').to(device)
        self.q22 = q().replace_var(x1='y2').to(device)
        self.q23 = q().replace_var(x1='y3').to(device)

        self.q31 = q().replace_var(x1='z1').to(device)
        self.q32 = q().replace_var(x1='z2').to(device)
        self.q33 = q().replace_var(x1='z3').to(device)

        self.q1 = ProductOfNormal([self.q11, self.q12, self.q13], name="q1").to(device)
        self.q2 = ProductOfNormal([self.q21, self.q22, self.q23], name="q1").to(device)
        self.q3 = ProductOfNormal([self.q31, self.q32, self.q33], name="q1").to(device)

        self.px = p().replace_var(x1='x1').to(device)
        self.py = p().replace_var(x1='y1').to(device)
        self.pz = p().replace_var(x1='z1').to(device)

        self.p = self.px * self.py * self.pz

        self.beta = Parameter("beta")
        self.is_supervised = Parameter("is_supervised")

        self.q = ProductOfNormal([self.q11, self.q22, self.q33], name="q_all").to(device)

        self.q_xy1 = ProductOfNormal([self.q11, self.q22, self.q13], name="q_xy1").to(device)
        self.q_xy2 = ProductOfNormal([self.q11, self.q22, self.q23], name="q_xy2").to(device)

        self.q_yz2 = ProductOfNormal([self.q21, self.q22, self.q33], name="q_yz2").to(device)
        self.q_yz3 = ProductOfNormal([self.q31, self.q22, self.q33], name="q_yz3").to(device)

        self.q_xz1 = ProductOfNormal([self.q11, self.q12, self.q33], name="q_xz1").to(device)
        self.q_xz3 = ProductOfNormal([self.q11, self.q32, self.q33], name="q_xz3").to(device)

        self.px_u = unsupervised_distr_no_var(self.px).to(device)
        self.py_u = unsupervised_distr_no_var(self.py).to(device)
        self.pz_u = unsupervised_distr_no_var(self.pz).to(device)

        self.q1_u = unsupervised_distr_no_var(self.q1).to(device)
        self.q2_u = unsupervised_distr_no_var(self.q2).to(device)
        self.q3_u = unsupervised_distr_no_var(self.q3).to(device)

        loss_supervised_recon = -(
                self.px.log_prob().expectation(self.q) +
                self.py.log_prob().expectation(self.q) +
                self.pz.log_prob().expectation(self.q) +

                self.px.log_prob().expectation(self.q_xy1) +
                self.px.log_prob().expectation(self.q_xy2) +
                self.py.log_prob().expectation(self.q_xy1) +
                self.py.log_prob().expectation(self.q_xy2) +
                self.pz.log_prob().expectation(self.q_xy1) +
                self.pz.log_prob().expectation(self.q_xy2) +

                self.px.log_prob().expectation(self.q_yz2) +
                self.px.log_prob().expectation(self.q_yz3) +
                self.py.log_prob().expectation(self.q_yz2) +
                self.py.log_prob().expectation(self.q_yz3) +
                self.pz.log_prob().expectation(self.q_yz2) +
                self.pz.log_prob().expectation(self.q_yz3) +

                self.px.log_prob().expectation(self.q_xz1) +
                self.px.log_prob().expectation(self.q_xz3) +
                self.py.log_prob().expectation(self.q_xz1) +
                self.py.log_prob().expectation(self.q_xz3) +
                self.pz.log_prob().expectation(self.q_xz1) +
                self.pz.log_prob().expectation(self.q_xz3)
        )

        loss_supervised_kl = self.beta*(
                KullbackLeibler(self.q, self.q_xy1) +
                KullbackLeibler(self.q, self.q_xy2) +

                KullbackLeibler(self.q, self.q_yz2) +
                KullbackLeibler(self.q, self.q_yz3) +

                KullbackLeibler(self.q, self.q_xz1) +
                KullbackLeibler(self.q, self.q_xz3) +

                KullbackLeibler(self.q_xy1, self.prior) +
                KullbackLeibler(self.q_xy2, self.prior) +

                KullbackLeibler(self.q_yz2, self.prior) +
                KullbackLeibler(self.q_yz3, self.prior) +

                KullbackLeibler(self.q_xz1, self.prior) +
                KullbackLeibler(self.q_xz3, self.prior) +

                KullbackLeibler(self.q1, self.q_xy1) +
                KullbackLeibler(self.q2, self.q_xy1) +
                KullbackLeibler(self.q1, self.q_xy2) +
                KullbackLeibler(self.q2, self.q_xy2) +

                KullbackLeibler(self.q2, self.q_yz2) +
                KullbackLeibler(self.q3, self.q_yz2) +
                KullbackLeibler(self.q2, self.q_yz3) +
                KullbackLeibler(self.q3, self.q_yz3) +

                KullbackLeibler(self.q1, self.q_xz1) +
                KullbackLeibler(self.q2, self.q_xz1) +
                KullbackLeibler(self.q1, self.q_xz3) +
                KullbackLeibler(self.q3, self.q_xz3)
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

                self.q11,
                self.q12,
                self.q13,

                self.q21,
                self.q22,
                self.q23,

                self.q31,
                self.q32,
                self.q33,
            ],
            optimizer=optim.Adam,
            optimizer_params=optimizer_params
        )

    def model_args(self, x, y, z, xu, yu, zu, beta=1.0):
        return {
            "x1": x, "y1": y, "z1": z,
            "x2": x, "y2": y, "z2": z,
            "x3": x, "y3": y, "z3": z,
            "x1_u": xu, "y1_u": yu, "z1_u": zu,
            "x2_u": xu, "y2_u": yu, "z2_u": zu,
            "x3_u": xu, "y3_u": yu, "z3_u": zu,
            "beta": beta,
        }

    def eval_args(self, x, y, z):
        return {
            "x1": x, "y1": y, "z1": z,
            "x2": x, "y2": y, "z2": z,
            "x3": x, "y3": y, "z3": z,
            "x1_u": x, "y1_u": y, "z1_u": z,
            "x2_u": x, "y2_u": y, "z2_u": z,
            "x3_u": x, "y3_u": y, "z3_u": z,
            "beta": 1.0,
        }

    def sample_z_from_x(self, x, sample_shape=None):
        varrs = {"x1": x, "x2": x, "x3": x}
        if sample_shape:
            return self.q1.sample(varrs, sample_shape=[sample_shape], return_all=False)
        return self.q1.sample(varrs, return_all=False)

    def sample_z_from_y(self, y, sample_shape=None):
        varrs = {"y1": y, "y2": y, "y3": y}
        if sample_shape:
            return self.q2.sample(varrs, sample_shape=[sample_shape], return_all=False)
        return self.q2.sample(varrs, return_all=False)

    def sample_z_from_z(self, z, sample_shape=None):
        varrs = {"z1": z, "z2": z, "z3": z}
        if sample_shape:
            return self.q3.sample(varrs, sample_shape=[sample_shape], return_all=False)
        return self.q3.sample(varrs, return_all=False)

    def sample_z_all(self, x, y, z, sample_shape=None):
        varrs = {
            "x1": x, "y1": y, "z1": z,
            "x2": x, "y2": y, "z2": z,
            "x3": x, "y3": y, "z3": z,
        }
        if sample_shape:
            return self.q.sample(varrs, sample_shape=[sample_shape], return_all=False)
        return self.q.sample(varrs, return_all=False)

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

                self.q11,
                self.q12,
                self.q13,

                self.q21,
                self.q22,
                self.q23,

                self.q31,
                self.q32,
                self.q33,
            ]:
            result += sum(p.numel() for p in m.parameters() if p.requires_grad)
        return result
