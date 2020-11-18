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

from models.SVAE_three import SVAEThree
from models.SVAE_three_star import SVAEThreeStar
from models.VAEVAE_three import VAEVAEThree
from models.VAEVAE_three_star import VAEVAEThreeStar

from experiments.mnist_split_three.MNIST_split_three_inference import InferenceJointAll, InferenceJoint, \
        InferenceX, GeneratorX, z_dim, MnistResNet


batch_size = 100
epochs = 200
annealing_epochs = 0
best_loss = 0

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

root = '.'
model_root = '/work1/svegal'


def crop_third(img):
    N_or = 28
    N = 27
    up = transforms.functional.crop(img, 0, 0, N/3, N_or)
    middle = transforms.functional.crop(img, N/3, 0, N/3, N_or)
    down = transforms.functional.crop(img, 2*N/3, 0, N/3, N_or)
    return up, middle, down


transform = transforms.Compose([
    transforms.Lambda(lambd=lambda x: [transforms.ToTensor()(i) for i in crop_third(x)] + [transforms.ToTensor()(x)]),
])
target_transform = transforms.Lambda(lambd=lambda y: torch.eye(10)[y])
kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}


def save_checkpoint(model_obj, is_best, model_name, folder=f'{model_root}/saved_models', filename='checkpoint_{}.pth.tar'):
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
    file_path = os.path.join(f'{model_root}/saved_models', filename.format(model_name))
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path)
        model_obj.model.distributions.load_state_dict(checkpoint['distributions'])


def generate_images(labels_dict, name):
    classes = list(range(10))
    N = 10
    N_rows = 15
    fig, ax = plt.subplots(nrows=len(classes) * N_rows, ncols=N, figsize=(N - 3, len(classes) * 2))
    for j in classes:
        for i in range(N):
            idx = j * N_rows
            for m in range(N_rows):
                ax[idx + m][i].imshow(labels_dict[j][i][m])
                ax[idx + m][i].set_xticks([], [])
                ax[idx + m][i].set_yticks([], [])
    plt.savefig(f'{root}/results/' + name)


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
    fig, ax = plt.subplots(nrows=N, ncols=2, figsize=(2, N))
    z_prior = model_obj.sample_prior(sample_shape=1000)
    z_prior['z'] = z_prior['z'].squeeze(1)

    im_top = model_obj.reconstruct_x(z_prior)
    im_middle = model_obj.reconstruct_y(z_prior)
    im_bottom = model_obj.reconstruct_z(z_prior)
    recon_full = torch.cat((im_top, im_middle, im_bottom), 2)

    output_resnet = model_resnet(recon_full)
    pred_full = output_resnet.argmax(dim=1, keepdim=True)

    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    ims_numpy = recon_full.detach().to("cpu").numpy()
    for i in range(N):
        ax[i][0].imshow(ims_numpy[i].reshape((27, 28)))
        ax[i][1].text(0.5 * (left + right), 0.5 * (bottom + top), pred_full[i].item(),
                      horizontalalignment='center',
                      verticalalignment='center',
                      transform=ax[i][1].transAxes, fontsize=20, color='black')
        for j in range(2):
            ax[i][j].set_xticks([], [])
            ax[i][j].set_yticks([], [])
    image_name = f'{root}/results/1000_samples_{keyword}_{model_obj.name}_{n_no_labels}.png'
    plt.savefig(image_name)


def evaluate(model_obj, data_loader, n_no_labels, name='latent_space'):
    labels = defaultdict(list)
    for x, y in data_loader:
        up_all, middle_all, down_all = x[0], x[1], x[2]
        for i in range(up_all.shape[0]):
            up, middle, down = up_all[i].to(device), middle_all[i].to(device), down_all[i].to(device)
            if is_full(labels):
                break

            z_up = model_obj.sample_z_from_x(up.unsqueeze(0), sample_shape=10)
            recon_up_up = model_obj.reconstruct_x(z_up).detach().to("cpu").numpy()
            recon_up_middle = model_obj.reconstruct_y(z_up).detach().to("cpu").numpy()
            recon_up_down = model_obj.reconstruct_z(z_up).detach().to("cpu").numpy()

            z_middle = model_obj.sample_z_from_y(down.unsqueeze(0), sample_shape=10)
            recon_middle_up = model_obj.reconstruct_x(z_middle).detach().to("cpu").numpy()
            recon_middle_middle = model_obj.reconstruct_y(z_middle).detach().to("cpu").numpy()
            recon_middle_down = model_obj.reconstruct_z(z_middle).detach().to("cpu").numpy()

            z_down = model_obj.sample_z_from_z(down.unsqueeze(0), sample_shape=10)
            recon_down_up = model_obj.reconstruct_x(z_down).detach().to("cpu").numpy()
            recon_down_middle = model_obj.reconstruct_y(z_down).detach().to("cpu").numpy()
            recon_down_down = model_obj.reconstruct_z(z_down).detach().to("cpu").numpy()

            z_full = model_obj.sample_z_all(up.unsqueeze(0), down.unsqueeze(0), sample_shape=10)
            recon_full_up = model_obj.reconstruct_x(z_full).detach().to("cpu").numpy()
            recon_full_middle = model_obj.reconstruct_y(z_full).detach().to("cpu").numpy()
            recon_full_down = model_obj.reconstruct_z(z_full).detach().to("cpu").numpy()

            labels[y[i].argmax().item()].append((
                up.detach().to("cpu").numpy()[0],
                middle.detach().to("cpu").numpy()[0],
                down.detach().to("cpu").numpy()[0],
                recon_up_up[0][0],
                recon_up_middle[0][0],
                recon_up_down[0][0],
                recon_middle_up[0][0],
                recon_middle_middle[0][0],
                recon_middle_down[0][0],
                recon_down_up[0][0],
                recon_down_middle[0][0],
                recon_down_down[0][0],
                recon_full_up[0][0],
                recon_full_middle[0][0],
                recon_full_down[0][0],
            ))
    image_name = f'{root}/results/{keyword}_{model_obj.name}_{n_no_labels}_{name}.png'
    generate_images(labels, image_name)


def evaluate_accuracy(model_obj, data_loader):
    correct_full, correct_up, correct_middle, correct_down = 0, 0, 0, 0
    total = 0
    for x, y in data_loader:
        up_all, middle_all, down_all = x[0].to(device), x[1].to(device), x[2].to(device)
        y = y.to(device)

        line = torch.zeros(up_all.size(0), 1, 28) # size correction

        z_up = model_obj.sample_z_from_x(up_all)
        im_top = model_obj.reconstruct_x(z_up)
        im_middle = model_obj.reconstruct_y(z_up)
        im_bottom = model_obj.reconstruct_z(z_up)
        recon_up = torch.cat((im_top, im_middle, im_bottom, line), 2)

        z_middle = model_obj.sample_z_from_y(middle_all)
        recon_middle_up = model_obj.reconstruct_x(z_middle)
        recon_middle_middle = model_obj.reconstruct_x(z_middle)
        recon_middle_down = model_obj.reconstruct_y(z_middle)
        recon_middle = torch.cat((recon_middle_up, recon_middle_middle, recon_middle_down, line), 2)

        z_down = model_obj.sample_z_from_z(down_all)
        recon_down_up = model_obj.reconstruct_x(z_down)
        recon_down_middle = model_obj.reconstruct_y(z_down)
        recon_down_down = model_obj.reconstruct_z(z_down)
        recon_down = torch.cat((recon_down_up, recon_down_middle, recon_down_down, line), 2)

        z_full = model_obj.sample_z_all(up_all, middle_all, down_all)
        recon_full_up = model_obj.reconstruct_x(z_full)
        recon_full_middle = model_obj.reconstruct_x(z_full)
        recon_full_down = model_obj.reconstruct_y(z_full)
        recon_full = torch.cat((recon_full_up, recon_full_middle, recon_full_down, line), 2)

        target = y.argmax(dim=1)

        output_up = model_resnet(recon_up)
        output_middle = model_resnet(recon_middle)
        output_down = model_resnet(recon_down)
        output_full = model_resnet(recon_full)

        pred_full = output_full.argmax(dim=1, keepdim=True)
        pred_up = output_up.argmax(dim=1, keepdim=True)
        pred_middle = output_middle.argmax(dim=1, keepdim=True)
        pred_down = output_down.argmax(dim=1, keepdim=True)

        correct_full += pred_full.eq(target.view_as(pred_full)).sum().item()
        correct_up += pred_up.eq(target.view_as(pred_up)).sum().item()
        correct_middle += pred_middle.eq(target.view_as(pred_middle)).sum().item()
        correct_down += pred_down.eq(target.view_as(pred_down)).sum().item()

        total += len(y)

    def accuracy(correct):
        return 100. * correct / total

    return accuracy(correct_full), accuracy(correct_up), accuracy(correct_middle), accuracy(correct_down)


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
        train_z_iterator = train_loader_unsupervised_z.__iter__()
        bsize = train_loader_unsupervised_y.batch_size
        dsize = train_loader_unsupervised_y.dataset
        for i in range(len(train_loader_unsupervised_y)):

            try:
                x, y = next(train_xy_iterator)
            except StopIteration:
                train_xy_iterator = train_loader_supervised.__iter__()
                x, y = next(train_xy_iterator)

            try:
                x_u, x_up_label = next(train_x_iterator)
            except StopIteration:
                train_x_iterator = train_loader_unsupervised_x.__iter__()
                x_u, x_up_label = next(train_x_iterator)

            try:
                y_u, y_down_label = next(train_y_iterator)
            except StopIteration:
                train_y_iterator = train_loader_unsupervised_y.__iter__()
                y_u, y_down_label = next(train_y_iterator)

            try:
                z_u, z_down_label = next(train_z_iterator)
            except StopIteration:
                train_z_iterator = train_loader_unsupervised_z.__iter__()
                z_u, z_down_label = next(train_z_iterator)

            beta = get_beta(epoch, i)
            current_beta = beta
            x_up, x_middle, x_down = x[0].to(device), x[1].to(device), x[2].to(device)
            x_up_split = x_u[0].to(device)
            x_middle_split = y_u[1].to(device)
            x_down_split = z_u[2].to(device)
            loss = model.train(model_obj.model_args(
                x_up,
                x_middle,
                x_down,
                x_up_split,
                x_middle_split,
                x_down_split,
                beta=beta
            ))
            train_loss += float(loss)
        train_loss = train_loss * bsize / len(dsize)
        return train_loss, current_beta

    def test():
        test_loss = 0
        test_xy_iterator = test_loader_supervised.__iter__()
        test_x_iterator = test_loader_unsupervised_x.__iter__()
        test_y_iterator = test_loader_unsupervised_y.__iter__()
        test_z_iterator = test_loader_unsupervised_z.__iter__()
        bsize = test_loader_unsupervised_y.batch_size
        dsize = test_loader_unsupervised_y.dataset
        for i in range(len(test_loader_unsupervised_y)):

            try:
                x, y = next(test_xy_iterator)
            except StopIteration:
                test_xy_iterator = test_loader_supervised.__iter__()
                x, y = next(test_xy_iterator)

            try:
                x_u, x_up_label = next(test_x_iterator)
            except StopIteration:
                test_x_iterator = test_loader_unsupervised_x.__iter__()
                x_u, x_up_label = next(test_x_iterator)

            try:
                y_u, y_down_label = next(test_y_iterator)
            except StopIteration:
                test_y_iterator = test_loader_unsupervised_y.__iter__()
                y_u, y_down_label = next(test_y_iterator)

            try:
                z_u, z_down_label = next(test_z_iterator)
            except StopIteration:
                test_z_iterator = test_loader_unsupervised_z.__iter__()
                z_u, z_down_label = next(test_z_iterator)

            x_up, x_middle, x_down = x[0].to(device), x[1].to(device), x[2].to(device)
            x_up_split = x_u[0].to(device)
            x_middle_split = y_u[1].to(device)
            x_down_split = z_u[2].to(device)
            loss = model.test(model_obj.model_args(
                x_up,
                x_middle,
                x_down,
                x_up_split,
                x_middle_split,
                x_down_split,
            ))
            test_loss += float(loss)
            result.extend([0] * x_up.shape[0])
        epoch_loss = test_loss * bsize / len(dsize)
        return epoch_loss

    train_losses, test_losses, accuracies_full, accuracies_up, accuracies_middle, accuracies_down = [], [], [], [], [], []
    accuracies_full_valid, accuracies_up_valid, accuracies_middle_valid, accuracies_down_valid = [], [], [], []
    for epoch in range(1, epochs+1):
        print('Model', model_obj.name, ', epoch', epoch, ', no labels', len(no_labels))
        loss, beta = train(epoch)
        train_losses.append(loss)
        ac_full, ac_up, ac_middle, ac_down = evaluate_accuracy(model_obj, test_loader_supervised)
        ac_full_valid, ac_up_valid, ac_middle_valid, ac_down_valid = evaluate_accuracy(model_obj, valid_loader_supervised)
        betas.append(beta)
        t_l = test()
        test_losses.append(t_l)
        is_best = ac_up > best_loss
        best_loss = max(ac_up, best_loss)
        save_checkpoint(model_obj, is_best, '{}_{}_{}'.format(model_obj.name, len(no_labels), keyword))
        accuracies_full.append(ac_full)
        accuracies_up.append(ac_up)
        accuracies_middle.append(ac_middle)
        accuracies_down.append(ac_down)

        accuracies_full_valid.append(ac_full_valid)
        accuracies_up_valid.append(ac_up_valid)
        accuracies_middle_valid.append(ac_middle_valid)
        accuracies_down_valid.append(ac_down_valid)

        result = pd.DataFrame({
            'train_loss': train_losses,
            'test_loss': test_losses,
            'n_parameters': [model_obj.get_number_of_parameters()] * len(train_losses),
            'accuracy_full': accuracies_full,
            'accuracy_up': accuracies_up,
            'accuracy_middle': accuracies_middle,
            'accuracy_down': accuracies_down,
            'accuracy_full_valid': accuracies_full_valid,
            'accuracy_up_valid': accuracies_up_valid,
            'accuracy_middle_valid': accuracies_middle_valid,
            'accuracy_down_valid': accuracies_down_valid,
        })
        result.to_csv(f'{root}/results/{model_obj.name}_{len(no_labels)}_{keyword}.csv')

    result = pd.DataFrame({
        'train_loss': train_losses,
        'test_loss': test_losses,
        'n_parameters': [model_obj.get_number_of_parameters()]*len(train_losses),
        'accuracy_full': accuracies_full,
        'accuracy_up': accuracies_up,
        'accuracy_middle': accuracies_middle,
        'accuracy_down': accuracies_down,
        'accuracy_full_valid': accuracies_full_valid,
        'accuracy_up_valid': accuracies_up_valid,
        'accuracy_middle_valid': accuracies_middle_valid,
        'accuracy_down_valid': accuracies_down_valid,
    })
    result.to_csv(f'{root}/results/{model_obj.name}_{len(no_labels)}_{keyword}.csv')


keyword = 'mnist_split_three'


def get_samplers(N_data, no_labels_share, index_start=0):
    indices = list(range(index_start, index_start+N_data))
    np.random.shuffle(indices)
    split = int(no_labels_share * N_data)
    train_idx, valid_idx_x = indices[split:], indices
    valid_idx_y = [i for i in valid_idx_x]
    valid_idx_z = [i for i in valid_idx_x]
    np.random.shuffle(valid_idx_y)
    np.random.shuffle(valid_idx_z)
    print(len(train_idx), len(valid_idx_x))
    unsupervised_sampler_x = SubsetRandomSampler(valid_idx_x)
    unsupervised_sampler_y = SubsetRandomSampler(valid_idx_y)
    unsupervised_sampler_z = SubsetRandomSampler(valid_idx_z)
    supervised_sampler = SubsetRandomSampler(train_idx)
    return unsupervised_sampler_x, unsupervised_sampler_y, unsupervised_sampler_z, supervised_sampler


N_data = 60000
no_labels_share = float(sys.argv[2])  # [0.0, 0.5, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999]
unsupervised_sampler_x, unsupervised_sampler_y, unsupervised_sampler_z, supervised_sampler = get_samplers(N_data, no_labels_share)

train_loader_full = torch.utils.data.DataLoader(
    datasets.MNIST(root=root, train=True, transform=transform, download=True, target_transform=target_transform),
    shuffle=True, **kwargs
)
train_loader_supervised = torch.utils.data.DataLoader(
    datasets.MNIST(root=root, train=True, transform=transform, download=True, target_transform=target_transform),
    sampler=supervised_sampler, **kwargs
)
train_loader_unsupervised_x = torch.utils.data.DataLoader(
    datasets.MNIST(root=root, train=True, transform=transform, download=True, target_transform=target_transform),
    sampler=unsupervised_sampler_x, **kwargs
)
train_loader_unsupervised_y = torch.utils.data.DataLoader(
    datasets.MNIST(root=root, train=True, transform=transform, download=True, target_transform=target_transform),
    sampler=unsupervised_sampler_y, **kwargs
)
train_loader_unsupervised_z = torch.utils.data.DataLoader(
    datasets.MNIST(root=root, train=True, transform=transform, download=True, target_transform=target_transform),
    sampler=unsupervised_sampler_z, **kwargs
)

N_test_data = 5000
unsupervised_sampler_x_test, unsupervised_sampler_y_test, unsupervised_sampler_z_test, supervised_sampler_test = get_samplers(N_test_data, 0)
test_loader_supervised = torch.utils.data.DataLoader(
    datasets.MNIST(root=root, train=False, transform=transform, download=True, target_transform=target_transform),
    sampler=supervised_sampler_test, **kwargs
)
test_loader_unsupervised_x = torch.utils.data.DataLoader(
    datasets.MNIST(root=root, train=False, transform=transform, download=True, target_transform=target_transform),
    sampler=unsupervised_sampler_x_test, **kwargs
)
test_loader_unsupervised_y = torch.utils.data.DataLoader(
    datasets.MNIST(root=root, train=False, transform=transform, download=True, target_transform=target_transform),
    sampler=unsupervised_sampler_y_test, **kwargs
)
test_loader_unsupervised_z = torch.utils.data.DataLoader(
    datasets.MNIST(root=root, train=False, transform=transform, download=True, target_transform=target_transform),
    sampler=unsupervised_sampler_z_test, **kwargs
)

N_valid_data = 5000
_, _, _, supervised_sampler_valid = get_samplers(N_valid_data, 0, index_start=5000)
valid_loader_supervised = torch.utils.data.DataLoader(
    datasets.MNIST(root=root, train=False, transform=transform, download=True, target_transform=target_transform),
    sampler=supervised_sampler_valid, **kwargs
)


models_classes = {
    'SVAEThree': SVAEThree,
    'VAEVAEThree': VAEVAEThree,
    'SVAEThreeStar': SVAEThreeStar,
    'VAEVAEThreeStar': VAEVAEThreeStar,
}
print(sys.argv)

N = int(N_data / batch_size)
model_class = models_classes[sys.argv[1]]


def init_model(model_class):
    return model_class(z_dim, {"lr": 2e-4}, InferenceX, GeneratorX, q_double=InferenceJoint, q_triple=InferenceJointAll)


model_obj = init_model(model_class)
filepath_saved_model = f'{model_root}/saved_models/mnist_cnn.pt'

model_resnet = MnistResNet().to(device)
model_resnet.load_state_dict(torch.load(filepath_saved_model))
no_labels_indices = set(np.random.choice(N, size=int(no_labels_share * N), replace=False))
if len(sys.argv) > 3 and sys.argv[3] == 'eval':
    is_best = not (len(sys.argv) > 4 and sys.argv[4] == 'current')
    name = '{}_{}_{}'.format(model_obj.name, len(no_labels_indices), keyword)
    load_checkpoint(model_obj, name, is_best=is_best)
    generate_1000_sampled_images(model_obj, keyword, len(no_labels_indices))
    evaluate(model_obj, test_loader_supervised, len(no_labels_indices), name=sys.argv[4])
else:
    name = '{}_{}_{}'.format(model_obj.name, len(no_labels_indices), keyword)
    load_checkpoint(model_obj, name, is_best=False)
    run_semisupervised(model_obj, no_labels_indices)
