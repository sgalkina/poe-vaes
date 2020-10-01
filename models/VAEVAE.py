from pixyz.distributions import Normal, ProductOfNormal
from pixyz.losses import KullbackLeibler, Parameter
from pixyz.models import Model
from torch import optim
import torch
from scipy.special import logsumexp
from models.utils import unsupervised_distr


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class VAEVAE(object):
    def __init__(self, z_dim, optimizer_params, q_x, q_y, p_x, p_y, q=None, q_star_y=None, q_star_x=None, extra_modules=None, y_coef=1):
        self.sample_shape = 1000
        self.z_dim = z_dim
        if extra_modules is None:
            extra_modules = []

        self.prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                       var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

        self.prior_both = Normal(loc=torch.tensor(0.), scale=torch.tensor(0.5),
                       var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

        self.p_x = p_x.to(device)
        self.p_y = p_y.to(device)
        self.p = self.p_x * self.p_y

        self.q_x = q_x.to(device)
        self.q_y = q_y.to(device)

        self.q = q.to(device)

        self.q1 = self.q_x.to(device)
        self.q2 = self.q_y.to(device)

        self.beta = Parameter("beta")
        self.is_supervised = Parameter("is_supervised")

        self.name = 'VAEVAE'

        self.p_x_u, self.x_vars = unsupervised_distr(self.p_x)
        self.p_y_u, self.y_vars = unsupervised_distr(self.p_y)
        self.q_x_u, self.x_q_vars = unsupervised_distr(self.q_x)
        self.q_y_u, self.y_q_vars = unsupervised_distr(self.q_y)

        _, self.xy_vars = unsupervised_distr(self.q)

        self.p_x_u = self.p_x_u.to(device)
        self.p_y_u = self.p_y_u.to(device)
        self.q_x_u = self.q_x_u.to(device)
        self.q_y_u = self.q_y_u.to(device)

        loss_supervised = -self.p_x.log_prob().expectation(self.q) - y_coef*self.p_y.log_prob().expectation(self.q) + \
                          self.beta * KullbackLeibler(self.q, self.q_x) + self.beta * KullbackLeibler(self.q, self.q_y)
        loss_unsupervised_x = -self.p_x_u.log_prob().expectation(self.q_x_u) + self.beta*KullbackLeibler(self.q_x_u, self.prior)
        loss_unsupervised_y = -y_coef*self.p_y_u.log_prob().expectation(self.q_y_u) + self.beta*KullbackLeibler(self.q_y_u, self.prior)

        self.model = Model(
            loss=loss_supervised.mean() + loss_unsupervised_y.mean() + loss_unsupervised_x.mean(),
            distributions=extra_modules + [self.p_x, self.p_y, self.q_x, self.q_y, self.q],
            optimizer=optim.Adam,
            optimizer_params=optimizer_params,
        )

    def model_args(self, x, y, xu, yu, beta=1.0, is_supervised=1.0):
        return {"x1": x, "y1": y, "beta": beta, "x1_u": xu, "y1_u": yu, 'is_supervised': is_supervised}

    def eval_args(self, x, y):
        return {"y1": y, "x1": x, "x1_u": x, "y1_u": y, "beta": 1.0}

    def loss_components(self, x, y, x_u, y_u):
        results = []
        for loss in [
            -self.p_x.log_prob().expectation(self.q).mean().eval({'x1': x, 'y1': y}),
            -self.p_y.log_prob().expectation(self.q).mean().eval({'x1': x, 'y1': y}),
            KullbackLeibler(self.q, self.q_x).mean().eval({'x1': x, 'y1': y}),
            KullbackLeibler(self.q, self.q_y).mean().eval({'x1': x, 'y1': y}),
            0,
            0,
            -self.p_x_u.log_prob().expectation(self.q_x_u).mean().eval({'x1_u': x_u}),
            KullbackLeibler(self.q_x_u, self.prior).mean().eval({'x1_u': x_u}),
            -self.p_y_u.log_prob().expectation(self.q_y_u).mean().eval({'y1_u': y_u}),
            KullbackLeibler(self.q_y_u, self.prior).mean().eval({'y1_u': y_u}),
        ]:
            results.append(float(loss))
        return results

    def log_likelihood_function(self, x, sample_size=1000):
        result = []
        for i in range(x.shape[0]):
            z = self.q1.sample({'x1': x[i].unsqueeze(0)}, sample_shape=[sample_size])['z']
            p_probs = self.prior.log_prob().eval({'z': z})

            a = self.q1.get_params({'x1': x[i].unsqueeze(0)})
            expert = Normal(loc=a['loc'], scale=a['scale'],
                            var=["z"], features_shape=[1, self.z_dim], name="expert").to(device)
            q_probs = expert.log_prob().eval({'z': z, 'x1': x[i].unsqueeze(0)})

            p_x_z_probs = self.p_x.log_prob().eval({'z': z, 'x1': x[i].unsqueeze(0)})
            a = logsumexp((p_x_z_probs + p_probs - q_probs).detach().cpu().numpy(), axis=-1)
            result.append(float(a))
        return result

    def test_joint_log_likelihood(self, x, y, sample_size=1000):
        result = []
        for i in range(x.shape[0]):
            z = self.q.sample({'x1': x[i].unsqueeze(0), 'y1': y[i].unsqueeze(0)}, sample_shape=[sample_size])['z']
            p_probs = self.prior.log_prob().eval({'z': z})
            p_x_z_probs = self.p_x.log_prob().eval({'z': z, 'x1': x[i].unsqueeze(0)})
            p_y_z_probs = self.p_y.log_prob().eval({'z': z, 'y1': y[i].unsqueeze(0)})

            a = self.q.get_params({'x1': x[i].unsqueeze(0), 'y1': y[i].unsqueeze(0)})
            expert = Normal(loc=a['loc'], scale=a['scale'],
                            var=["z"], features_shape=[1, self.z_dim], name="expert").to(device)
            q_probs = expert.log_prob().eval({'z': z, 'x1': x[i].unsqueeze(0), 'y1': y[i].unsqueeze(0)})

            a = logsumexp((p_x_z_probs + p_y_z_probs + p_probs - q_probs).detach().cpu().numpy(), axis=-1)
            result.append(float(a))
        return result

    def log_likelihood_function_x_y(self, x, y, sample_size=1000):
        result = []
        for i in range(x.shape[0]):
            z = self.q.sample({'x1': x[i].unsqueeze(0), 'y1': y[i].unsqueeze(0)}, sample_shape=[sample_size])['z']
            p_probs = self.prior.log_prob().eval({'z': z})
            p_x_z_probs = self.p_x.log_prob().eval({'z': z, 'x1': x[i].unsqueeze(0)})

            a = self.q.get_params({'x1': x[i].unsqueeze(0), 'y1': y[i].unsqueeze(0)})
            expert = Normal(loc=a['loc'], scale=a['scale'],
                            var=["z"], features_shape=[1, self.z_dim], name="expert").to(device)
            q_probs = expert.log_prob().eval({'z': z, 'x1': x[i].unsqueeze(0), 'y1': y[i].unsqueeze(0)})

            a = logsumexp((p_x_z_probs + p_probs - q_probs).detach().cpu().numpy(), axis=-1)
            result.append(float(a))
        return result

    def sample_z(self, y, sample_shape=None):
        if sample_shape:
            return self.q_y.sample({"y1": y}, sample_shape=[sample_shape], return_all=False)
        return self.q_y.sample({"y1": y}, return_all=False)

    def sample_z_from_x(self, x, sample_shape=None):
        if sample_shape:
            return self.q_x.sample({"x1": x}, sample_shape=[sample_shape], return_all=False)
        return self.q_x.sample({"x1": x}, return_all=False)

    def sample_z_all(self, x, y, sample_shape=None):
        if sample_shape:
            return self.q.sample({"x1": x, "y1": y}, sample_shape=[sample_shape], return_all=False)
        return self.q.sample({"x1": x, "y1": y}, return_all=False)

    def reconstruct_x(self, z):
        return self.p_x.sample_mean(z)

    def reconstruct_y(self, z):
        return self.p_y.sample_mean(z)

    def sample_prior(self, sample_shape=None):
        if sample_shape:
            return self.prior.sample(sample_shape=[sample_shape], return_all=False)
        return self.prior.sample(return_all=False)

    def get_number_of_parameters(self):
        result = 0
        for m in [self.q_x, self.q_y]:
            result += sum(p.numel() for p in m.parameters() if p.requires_grad)
        return result