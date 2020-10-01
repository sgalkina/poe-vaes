import matplotlib
matplotlib.use('Agg')
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from collections import defaultdict
import pandas as pd
import os
import sys
import shutil
import matplotlib.pyplot as plt
from torchnet.dataset import TensorDataset, ResampleDataset

from models.VAEVAE import VAEVAE
from models.SVAE import SVAE
from models.VAEVAE_star import VAEVAE_star as VAEVAE_star
from models.SVAE_star import SVAE_star as SVAE_star

from experiments.mnist_svhn.MNIST_SVHN_inference import GeneratorX, GeneratorY, \
    InferenceX, InferenceX_missing, InferenceY, InferenceY_missing, z_dim, \
    InferenceJoint, SVHN_Classifier, MNIST_Classifier

batch_size = 256
epochs = 200
annealing_epochs = 0
best_loss = 0

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

root = "misc"

kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}


def save_checkpoint(model_obj, is_best, model_name, folder=f'{root}/saved_models', filename='checkpoint_{}.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(dict(distributions=model_obj.model.distributions.state_dict()), os.path.join(folder, filename.format(model_name)))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename.format(model_name)),
                        os.path.join(folder, 'model_best_{}.pth.tar'.format(model_name)))


def load_checkpoint(model_obj, model_name, is_best=True):
    if is_best:
        filename = 'model_best_{}.pth.tar'
    else:
        filename = 'checkpoint_{}.pth.tar'
    file_path = os.path.join(f'{root}/saved_models', filename.format(model_name))
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path)
        model_obj.model.distributions.load_state_dict(checkpoint['distributions'])


def generate_images(labels_dict, name):
    classes = list(range(10))
    N = 10
    N_rows = 8
    fig, ax = plt.subplots(nrows=len(classes) * N_rows, ncols=N, figsize=(N - 3, len(classes) * 2))
    for j in classes:
        for i in range(N):
            idx = j * N_rows
            for m in range(N_rows):
                ax[idx + m][i].imshow(labels_dict[j][i][m])
                ax[idx + m][i].set_xticks([], [])
                ax[idx + m][i].set_yticks([], [])
    plt.savefig('results/' + name)


def is_full(labels_storage):
    N_labels = 10
    if len(labels_storage) < N_labels:
        return False
    for k, v in labels_storage.items():
        if len(v) < N_labels:
            return False
    return True


def generate_1000_sampled_images(model_obj, keyword, n_no_labels):
    N = 30
    z_prior = model_obj.sample_prior(sample_shape=1000)
    z_prior['z'] = z_prior['z'].squeeze(1)

    im_mnist = model_obj.reconstruct_x(z_prior)
    im_svhn = model_obj.reconstruct_y(z_prior)

    mnist_mnist = mnist_net(im_mnist)
    svhn_svhn = svhn_net(im_svhn)

    _, pred_m = torch.max(mnist_mnist.data, 1)
    _, pred_s = torch.max(svhn_svhn.data, 1)

    preds = [pred_m, pred_s]

    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    shapes = [(1000, 28, 28), (1000, 3, 32, 32)]
    for p, im in enumerate([
        im_mnist.detach().to("cpu").numpy().reshape(shapes[0]),
        im_svhn.detach().to("cpu").numpy().reshape(shapes[1]).transpose((0, 2, 3, 1)),
    ]):
        fig, ax = plt.subplots(nrows=N, ncols=2, figsize=(2, N))
        for i in range(N):
            ax[i][0].imshow(im[i])
            ax[i][1].text(0.5 * (left + right), 0.5 * (bottom + top), preds[p][i].item(),
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=ax[i][1].transAxes, fontsize=20, color='black')
            for j in range(2):
                ax[i][j].set_xticks([], [])
                ax[i][j].set_yticks([], [])
        image_name = f'{root}/results/1000_samples_{keyword}_{model_obj.name}_{n_no_labels}_{p}.png'
        plt.savefig(image_name)


def evaluate(model_obj, data_loader, n_no_labels, name='latent_space'):
    labels = defaultdict(list)
    for x, y in data_loader:
        up_all, down_all = x[0], x[1]
        for i in range(up_all.shape[0]):
            up, down = up_all[i].to(device), down_all[i].to(device)
            if is_full(labels):
                break

            z_up = model_obj.sample_z_from_x(up.unsqueeze(0), sample_shape=10)
            recon_up_up = model_obj.reconstruct_x(z_up).detach().to("cpu").numpy()
            recon_up_down = model_obj.reconstruct_y(z_up).detach().to("cpu").numpy()

            z_down = model_obj.sample_z(down.unsqueeze(0), sample_shape=10)
            recon_down_up = model_obj.reconstruct_x(z_down).detach().to("cpu").numpy()
            recon_down_down = model_obj.reconstruct_y(z_down).detach().to("cpu").numpy()

            z_full = model_obj.sample_z_all(up.unsqueeze(0), down.unsqueeze(0), sample_shape=10)
            recon_full_up = model_obj.reconstruct_x(z_full).detach().to("cpu").numpy()
            recon_full_down = model_obj.reconstruct_y(z_full).detach().to("cpu").numpy()

            labels[y[i].argmax().item()].append((
                up.detach().to("cpu").numpy()[0],
                down.detach().to("cpu").numpy()[0],
                recon_up_up[0][0],
                recon_up_down[0][0],
                recon_down_up[0][0],
                recon_down_down[0][0],
                recon_full_up[0][0],
                recon_full_down[0][0],
            ))
    image_name = f'{root}/results/{keyword}_{model_obj.name}_{n_no_labels}_{name}.png'
    generate_images(labels, image_name)


def joint_coherence(model_obj):
    total = 0
    corr = 0

    z_prior = model_obj.sample_prior(sample_shape=10000)
    z_prior['z'] = z_prior['z'].squeeze(1)
    mnist = model_obj.reconstruct_x(z_prior)
    svhn = model_obj.reconstruct_y(z_prior)

    mnist_mnist = mnist_net(mnist)
    svhn_svhn = svhn_net(svhn)

    _, pred_m = torch.max(mnist_mnist.data, 1)
    _, pred_s = torch.max(svhn_svhn.data, 1)
    total += pred_m.size(0)
    corr += (pred_m == pred_s).sum().item()

    print('Joint coherence: {:.2f}%'.format(corr / total * 100))
    return corr / total


def evaluate_accuracy(model_obj, data_loader):
    correct_up, correct_down = 0, 0
    total = 0
    for x, y in data_loader:
        mnist_all, svhn_all = x[0].to(device), y[0].to(device)
        y = x[1].to(device)

        z_mnist = model_obj.sample_z_from_x(mnist_all)
        recon_svhn_from_mnist = model_obj.reconstruct_y(z_mnist)

        z_svhn = model_obj.sample_z(svhn_all)
        recon_mnist_from_svhn = model_obj.reconstruct_x(z_svhn)

        target = y

        output_mnist = mnist_net(recon_mnist_from_svhn)
        output_svhn = svhn_net(recon_svhn_from_mnist)

        _, pred_m = torch.max(output_mnist.data, 1)
        _, pred_s = torch.max(output_svhn.data, 1)

        total += pred_m.size(0)

        correct_up += (pred_m == target).sum().item()
        correct_down += (pred_s == target).sum().item()

    return correct_up / total, correct_down / total


def evaluate_log_lik(model_obj, data_loader):
    log_p_x, log_p_x_y = [], []
    for x, y in data_loader:
        up_all, down_all = x[0], y[0]
        x = up_all.to(device)
        y = down_all.to(device)
        log_p_x_y.extend(model_obj.log_likelihood_function_x_y(x, y))
        log_p_x.extend(model_obj.log_likelihood_function(x))
    return np.mean(log_p_x), np.mean(log_p_x_y)


def get_beta(epoch, i):
    if epoch < annealing_epochs:
        N_mini_batches = int(N_data / batch_size)
        return float(i + (epoch - 1) * N_mini_batches + 1) / float(annealing_epochs * N_mini_batches)
    else:
        return 1.0


current_beta = 0


def run_semisupervised(model_obj, no_labels):
    global best_loss, current_beta

    betas = []
    model = model_obj.model

    def train(epoch):
        global current_beta
        train_loss = 0
        train_xy_iterator = train_loader_supervised.__iter__()
        train_x_iterator = train_loader_unsupervised_x.__iter__()
        train_y_iterator = train_loader_unsupervised_y.__iter__()
        total = 0
        if model_obj.name == 'JVAEVAE':
            bsize = train_loader_supervised.batch_size
            dsize = train_loader_supervised.dataset
            for i, (x, y) in enumerate(train_loader_supervised):
                beta = get_beta(epoch, i)
                current_beta = beta
                x_mnist, y_svhn = x[0].to(device), y[0].to(device)
                loss = model.train(model_obj.model_args(x_mnist, y_svhn, x_mnist, y_svhn, beta=beta))
                train_loss += float(loss)
            train_loss = train_loss * bsize / len(dsize)
        else:
            bsize = train_loader_unsupervised_y.batch_size
            dsize = train_loader_unsupervised_y.dataset
            for i in range(len(train_loader_unsupervised_y)):

                try:
                    x, y = next(train_xy_iterator)
                except StopIteration:
                    train_xy_iterator = train_loader_supervised.__iter__()
                    x, y = next(train_xy_iterator)

                try:
                    x_u, _ = next(train_x_iterator)
                except StopIteration:
                    train_x_iterator = train_loader_unsupervised_x.__iter__()
                    x_u, _ = next(train_x_iterator)

                try:
                    _, y_u = next(train_y_iterator)
                except StopIteration:
                    train_y_iterator = train_loader_unsupervised_y.__iter__()
                    _, y_u = next(train_y_iterator)

                beta = get_beta(epoch, i)
                current_beta = beta
                x_mnist, y_svhn = x[0].to(device), y[0].to(device)
                x_mnist_unsup = x_u[0].to(device)
                y_svhn_unsup = y_u[0].to(device)
                loss = model.train(model_obj.model_args(
                    x_mnist,
                    y_svhn,
                    x_mnist_unsup,
                    y_svhn_unsup,
                    beta=beta
                ))
                train_loss += float(loss)
                total += 1
                print(f'Batch {i} out of {len(train_loader_unsupervised_y)}', flush=True)
            train_loss = train_loss * bsize / len(dsize)
        return train_loss, current_beta

    def test():
        test_loss = 0
        test_xy_iterator = test_loader_supervised.__iter__()
        test_x_iterator = test_loader_unsupervised_x.__iter__()
        test_y_iterator = test_loader_unsupervised_y.__iter__()
        total = 0
        for i in range(len(test_loader_unsupervised_y)):

            try:
                x, y = next(test_xy_iterator)
            except StopIteration:
                test_xy_iterator = test_loader_supervised.__iter__()
                x, y = next(test_xy_iterator)

            try:
                x_u, _ = next(test_x_iterator)
            except StopIteration:
                test_x_iterator = test_loader_unsupervised_x.__iter__()
                x_u, _ = next(test_x_iterator)

            try:
                _, y_u = next(test_y_iterator)
            except StopIteration:
                test_y_iterator = test_loader_unsupervised_y.__iter__()
                _, y_u = next(test_y_iterator)

            x_mnist, y_svhn = x[0].to(device), y[0].to(device)
            x_mnist_unsup = x_u[0].to(device)
            y_svhn_unsup = y_u[0].to(device)
            loss = model.test(model_obj.model_args(
                x_mnist,
                y_svhn,
                x_mnist_unsup,
                y_svhn_unsup,
            ))
            test_loss += float(loss)
            total += 1
        epoch_loss = test_loss / total
        return epoch_loss

    train_losses, test_losses, accuracies_joint, accuracies_s2m, accuracies_m2s = [], [], [], [], []

    for epoch in range(1, epochs+1):
        print('Model', model_obj.name, ', epoch', epoch, ', no labels', len(no_labels))
        loss, beta = train(epoch)
        train_losses.append(loss)

        ac_s2m, ac_m2s = evaluate_accuracy(model_obj, test_loader_full)
        ac_joint = joint_coherence(model_obj)

        betas.append(beta)
        t_l = test()
        is_best = ac_joint > best_loss
        best_loss = max(ac_joint, best_loss)
        save_checkpoint(model_obj, is_best, '{}_{}_{}'.format(model_obj.name, len(no_labels), keyword))
        test_losses.append(t_l)

        accuracies_joint.append(ac_joint)
        accuracies_s2m.append(ac_s2m)
        accuracies_m2s.append(ac_m2s)

        result = pd.DataFrame({
            'train_loss': train_losses,
            'test_loss': test_losses,
            'n_parameters': [model_obj.get_number_of_parameters()] * len(train_losses),
            'accuracy_joint': accuracies_joint,
            'accuracy_s2m': accuracies_s2m,
            'accuracy_m2s': accuracies_m2s,
        })
        result.to_csv(f'{root}/results/{model_obj.name}_{len(no_labels)}_{keyword}.csv', index=False)

    result = pd.DataFrame({
        'train_loss': train_losses,
        'test_loss': test_losses,
        'n_parameters': [model_obj.get_number_of_parameters()]*len(train_losses),
        'accuracy_joint': accuracies_joint,
        'accuracy_s2m': accuracies_s2m,
        'accuracy_m2s': accuracies_m2s,
    })
    result.to_csv(f'{root}/results/{model_obj.name}_{len(no_labels)}_{keyword}.csv', index=False)


keyword = 'mnist_svhn'


def get_samplers(N_data, no_labels_share, index_start=0):
    indices = list(range(index_start, index_start+N_data))
    np.random.seed(1)
    np.random.shuffle(indices)
    split = int(no_labels_share * N_data)
    train_idx, valid_idx_x = indices[split:], indices
    valid_idx_y = [i for i in valid_idx_x]
    np.random.shuffle(valid_idx_y)
    print(len(train_idx), len(valid_idx_x))
    unsupervised_sampler_x = SubsetRandomSampler(valid_idx_x)
    unsupervised_sampler_y = SubsetRandomSampler(valid_idx_y)
    supervised_sampler = SubsetRandomSampler(train_idx)
    return unsupervised_sampler_x, unsupervised_sampler_y, supervised_sampler


# get transformed indices
t_mnist = torch.load(f'{root}/saved_models/train-ms-mnist-idx.pt')
t_svhn = torch.load(f'{root}/saved_models/train-ms-svhn-idx.pt')
s_mnist = torch.load(f'{root}/saved_models/test-ms-mnist-idx.pt')
s_svhn = torch.load(f'{root}/saved_models/test-ms-svhn-idx.pt')

# load base datasets

tx = transforms.Compose([transforms.ToTensor()])
t1 = torch.utils.data.DataLoader(datasets.MNIST(root, train=True, download=True, transform=tx),
                shuffle=True, **kwargs)
s1 = torch.utils.data.DataLoader(datasets.MNIST(root, train=False, download=True, transform=tx),
                shuffle=True, **kwargs)

t2 = torch.utils.data.DataLoader(datasets.SVHN(root, split='train', download=True, transform=tx),
                shuffle=True, **kwargs)
s2 = torch.utils.data.DataLoader(datasets.SVHN(root, split='test', download=True, transform=tx),
                shuffle=True, **kwargs)

train_mnist_svhn = TensorDataset([
    ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len(t_mnist)),
    ResampleDataset(t2.dataset, lambda d, i: t_svhn[i], size=len(t_svhn))
])
test_mnist_svhn = TensorDataset([
    ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
    ResampleDataset(s2.dataset, lambda d, i: s_svhn[i], size=len(s_svhn))
])

N_data = 1682040
no_labels_share = float(sys.argv[2])  # [0.0, 0.5, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999]
unsupervised_sampler_x, unsupervised_sampler_y, supervised_sampler = get_samplers(N_data, no_labels_share)

train_loader_full = torch.utils.data.DataLoader(train_mnist_svhn, shuffle=True, **kwargs)
train_loader_supervised = torch.utils.data.DataLoader(train_mnist_svhn, sampler=supervised_sampler, **kwargs)
train_loader_unsupervised_x = torch.utils.data.DataLoader(train_mnist_svhn, sampler=unsupervised_sampler_x, **kwargs)
train_loader_unsupervised_y = torch.utils.data.DataLoader(train_mnist_svhn, sampler=unsupervised_sampler_y, **kwargs)

N_test_data = 300000
unsupervised_sampler_x_test, unsupervised_sampler_y_test, supervised_sampler_test = get_samplers(N_test_data, no_labels_share)

test_loader_full = torch.utils.data.DataLoader(test_mnist_svhn, shuffle=True, **kwargs)
test_loader_supervised = torch.utils.data.DataLoader(test_mnist_svhn, sampler=supervised_sampler_test, **kwargs)
test_loader_unsupervised_x = torch.utils.data.DataLoader(test_mnist_svhn, sampler=unsupervised_sampler_x_test, **kwargs)
test_loader_unsupervised_y = torch.utils.data.DataLoader(test_mnist_svhn, sampler=unsupervised_sampler_y_test, **kwargs)


models_classes = {
    'SVAE': SVAE,
    'SVAE_star': SVAE_star,
    'VAEVAE': VAEVAE,
    'VAEVAE_star': VAEVAE_star,
}
print(sys.argv)

N = int(N_data / batch_size)
model_class = models_classes[sys.argv[1]]


def init_model(model_class):
    p_x = GeneratorX()
    p_y = GeneratorY()

    q_x = InferenceX()
    q_y = InferenceY()

    q_star_y = InferenceY_missing()
    q_star_x = InferenceX_missing()

    q = InferenceJoint()

    return model_class(z_dim, {"lr": 1e-4}, q_x, q_y, p_x, p_y, q=q, q_star_y=q_star_y, q_star_x=q_star_x)


model_obj = init_model(model_class)

mnist_net, svhn_net = MNIST_Classifier().to(device), SVHN_Classifier().to(device)
mnist_net.load_state_dict(torch.load(f'{root}/saved_models/mnist_model_MVAEVAE.pt'))
svhn_net.load_state_dict(torch.load(f'{root}/saved_models/svhn_model_MVAEVAE.pt'))

mnist_net.eval()
svhn_net.eval()

no_labels_indices = set(np.random.choice(N, size=int(no_labels_share * N), replace=False))
if len(sys.argv) > 3 and sys.argv[3] == 'eval':
    is_best = not (len(sys.argv) > 4 and sys.argv[4] == 'current')
    name = '{}_{}_{}'.format(model_obj.name, len(no_labels_indices), keyword)
    load_checkpoint(model_obj, name, is_best=is_best)
    generate_1000_sampled_images(model_obj, keyword, len(no_labels_indices))
else:
    name = '{}_{}_{}'.format(model_obj.name, len(no_labels_indices), keyword)
    load_checkpoint(model_obj, name, is_best=False)
    run_semisupervised(model_obj, no_labels_indices)
