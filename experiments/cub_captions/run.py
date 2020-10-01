import matplotlib
matplotlib.use('Agg')
import torch
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
import pandas as pd
import os
import numpy as np
import sys
import shutil
import matplotlib.pyplot as plt
from torchnet.dataset import TensorDataset, ResampleDataset
import json
import torch.nn.functional as F
import pickle

from models.VAEVAE import VAEVAE
from models.SVAE import SVAE
from models.SVAE_star import SVAE_star as SVAE_star
from models.VAEVAE_star import VAEVAE_star as VAEVAE_star

from experiments.cub_captions.cub_inference import GeneratorX, GeneratorY, \
    InferenceX, InferenceX_missing, InferenceY, InferenceY_missing, InferenceJoint, latentDim, maxSentLen, \
    vocabSize, lenWindow, minOccur

from experiments.cub_captions.datasets import CUBSentences, CUBImageFt
from experiments.cub_captions.helper import fetch_emb, fetch_weights, apply_weights, apply_pc


batch_size = 100
epochs = 300
annealing_epochs = 0
# seed = 1
# torch.manual_seed(seed)
best_loss = -100

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}


root = "misc"
path_current = f'{root}/oc:3_msl:32'
vocab_path = f'{path_current}/cub.vocab'
emb_path = f'{path_current}/cub.emb'
weights_path = f'{path_current}/cub.weights'
pc_path = f'{path_current}/cub.pc'
path_models = f'{root}'


def save_checkpoint(model_obj, is_best, model_name, folder=f'{path_models}/saved_models', filename='checkpoint_{}.pth.tar'):
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
    file_path = os.path.join(f'{path_models}/saved_models', filename.format(model_name))
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path)
        model_obj.model.distributions.load_state_dict(checkpoint['distributions'])


def load_vocab(self):
    # call dataloader function to create vocab file
    if not os.path.exists(self.vocab_file):
        _, _ = self.getDataLoaders(256)
    with open(self.vocab_file, 'r') as vocab_file:
        vocab = json.load(vocab_file)
    return vocab['i2w']


def _imshow(image, caption, i, fig, N):
    """Imshow for Tensor."""
    ax = fig.add_subplot(N // 2, 4, i * 2 + 1)
    ax.axis('off')
    image = image.cpu().detach().numpy().transpose((1, 2, 0))  #
    plt.imshow(image)
    ax = fig.add_subplot(N // 2, 4, i * 2 + 2)
    pos = ax.get_position()
    ax.axis('off')
    with open(vocab_path, 'r') as vocab_file:
        vocab = json.load(vocab_file)
        i2w = vocab['i2w']
    plt.text(
        x=0.5 * (pos.x0 + pos.x1),
        y=0.5 * (pos.y0 + pos.y1),
        ha='left',
        s='{}'.format(
            ' '.join(i2w[str(i)] + '\n' if (n + 1) % 5 == 0
                     else i2w[str(i)] for n, i in enumerate(caption))),
        fontsize=6,
        verticalalignment='center',
        horizontalalignment='center'
    )
    return fig


fn_trun = lambda s: s[:np.where(s == 2)[0][0] + 1] if 2 in s else s


def _sent_preprocess(sentences):
    """make sure raw data is always passed as dim=2 to avoid argmax.
    last dimension must always be word embedding."""
    if len(sentences.shape) > 2:
        sentences = sentences.argmax(-1).squeeze()
    return [fn_trun(s.cpu().detach().numpy()) for s in sentences]


def generate_images(keyword, images, captions):
    N = 8
    images = unproject(images, search_split='train')
    captions = _sent_preprocess(captions)
    fig = plt.figure(figsize=(8, 6))
    for i, (image, caption) in enumerate(zip(images, captions)):
        fig = _imshow(image, caption, i, fig, N)
    plt.savefig(f'{root}/results/results/gen_samples_{keyword}.png')
    plt.close()


def corrs_generated(im, sent):
    RESET = False
    emb = fetch_emb(lenWindow, minOccur, emb_path, vocab_path, RESET)
    weights = fetch_weights(weights_path, vocab_path, RESET, a=1e-3)
    emb = torch.from_numpy(emb).to(device)
    weights = torch.from_numpy(weights).to(device).type(emb.dtype)
    with open(pc_path, 'rb') as file:
        u = pickle.load(file).to(device)
    fn_to_emb = lambda data, emb=emb, weights=weights, u=u: \
        apply_pc(apply_weights(emb, weights, data), u)
    return calculate_corr(im, fn_to_emb(sent))


def qualitative_eval(model_obj, data_loader, no_labels, keyword_run=''):
    z_prior = model_obj.sample_prior(sample_shape=1000)
    z_prior['z'] = z_prior['z'].squeeze(1)
    gen_images = model_obj.reconstruct_y(z_prior)
    gen_sentences = model_obj.reconstruct_x(z_prior).argmax(axis=2)
    generate_images(f'{root}/results/eval_1000_samples_{keyword_run}_{keyword}_{len(no_labels)}_{model_obj.name}', gen_images[:8], gen_sentences[:8])
    print('Joint coherence for generated: ', corrs_generated(gen_images, gen_sentences))

    for i, (x, y) in enumerate(data_loader):
        im, sent = x.to(device), torch.eye(vocabSize)[y[0].long()].to(device)
        generate_images(f'{root}/results/eval_true_{keyword_run}_{keyword}_{len(no_labels)}_{model_obj.name}', im[:8], sent[:8])

        z_full = model_obj.sample_z_from_x(sent)
        recon_sent = model_obj.reconstruct_x(z_full)
        recon_im = model_obj.reconstruct_y(z_full)
        generate_images(f'{root}/results/eval_from_caption_{keyword_run}_{keyword}_{len(no_labels)}_{model_obj.name}', recon_im[:8], recon_sent[:8])

        z_full = model_obj.sample_z(im)
        recon_sent = model_obj.reconstruct_x(z_full)
        recon_im = model_obj.reconstruct_y(z_full)
        generate_images(f'{root}/results/eval_from_image_{keyword_run}_{keyword}_{len(no_labels)}_{model_obj.name}', recon_im[:8], recon_sent[:8])

        z_full = model_obj.sample_z_all(sent, im)
        recon_sent = model_obj.reconstruct_x(z_full)
        recon_im = model_obj.reconstruct_y(z_full)
        generate_images(f'{root}/results/eval_from_both_{keyword_run}_{keyword}_{len(no_labels)}_{model_obj.name}', recon_im[:8], recon_sent[:8])
        return


def calculate_corr(images, embeddings):
    im_mean = torch.load(path_current + '/images_mean.pt').to(device)
    emb_mean = torch.load(path_current + '/emb_mean.pt').to(device)
    im_proj = torch.load(path_current + '/im_proj.pt').to(device)
    emb_proj = torch.load(path_current + '/emb_proj.pt').to(device)
    with torch.no_grad():
        corr = F.cosine_similarity((images - im_mean) @ im_proj,
                                   (embeddings - emb_mean) @ emb_proj).mean()
    return corr


def quantitative_eval(model_obj, data_loader):
    RESET = False
    emb = fetch_emb(lenWindow, minOccur, emb_path, vocab_path, RESET)
    weights = fetch_weights(weights_path, vocab_path, RESET, a=1e-3)
    emb = torch.from_numpy(emb).to(device)
    weights = torch.from_numpy(weights).to(device).type(emb.dtype)
    with open(pc_path, 'rb') as file:
        u = pickle.load(file).to(device)
    fn_to_emb = lambda data, emb=emb, weights=weights, u=u: \
        apply_pc(apply_weights(emb, weights, data), u)
    i2t = []
    s2i = []
    gt = []
    for i, (x, y) in enumerate(data_loader):
        im, sent = x.to(device), y[0].to(device)
        sent_inp = torch.eye(vocabSize)[y[0].long()].to(device)
        z_full = model_obj.sample_z(im)
        recon_sent = model_obj.reconstruct_x(z_full).argmax(axis=2)

        z_full = model_obj.sample_z_from_x(sent_inp)
        recon_im = model_obj.reconstruct_y(z_full)

        i2t.append(calculate_corr(im, fn_to_emb(recon_sent)))
        s2i.append(calculate_corr(recon_im, fn_to_emb(sent.int())))
        gt.append(calculate_corr(im, fn_to_emb(sent.int())))

    z_prior = model_obj.sample_prior(sample_shape=1000)
    z_prior['z'] = z_prior['z'].squeeze(1)
    gen_images = model_obj.reconstruct_y(z_prior)
    gen_sentences = model_obj.reconstruct_x(z_prior).argmax(axis=2)

    joint_coherence = calculate_corr(gen_images, fn_to_emb(gen_sentences))
    cross_coherence_i2s = sum(i2t) / len(gt)
    cross_coherence_s2i = sum(s2i) / len(gt)
    cross_coherence_truth = sum(gt) / len(gt)
    return cross_coherence_truth.item(), cross_coherence_i2s.item(), cross_coherence_s2i.item(), joint_coherence.item()


def get_beta(epoch, i):
    if epoch < annealing_epochs:
        N_mini_batches = int(N_data / batch_size)
        return float(i + (epoch - 1) * N_mini_batches + 1) / float(annealing_epochs * N_mini_batches)
    else:
        return 1.0


current_beta = 0


def run_semisupervised(model_obj, no_labels, is_continue=False):
    global best_loss, current_beta

    betas = []
    model = model_obj.model

    def train(epoch):
        global current_beta
        train_loss = 0
        train_xy_iterator = train_loader_supervised.__iter__()
        train_x_iterator = train_loader_unsupervised_x.__iter__()
        train_y_iterator = train_loader_unsupervised_y.__iter__()
        components = [0] * len(loss_components)
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
            x_up, x_down = x.to(device), torch.eye(vocabSize)[y[0].long()].to(device)
            x_up_split = x_u.to(device)
            x_down_split = torch.eye(vocabSize)[y_u[0].long()].to(device)
            loss = model.train(model_obj.model_args(
                x_down,
                x_up,
                x_down_split,
                x_up_split,
                beta=beta
            ))

            train_loss += float(loss)
            for i, l in enumerate(model_obj.loss_components(
                    x_down,
                    x_up,
                    x_down_split,
                    x_up_split,
            )):
                components[i] += l
        for i, l in enumerate(components):
            loss_components_train[i].append(l * bsize / len(dsize))

        train_loss = train_loss * bsize / len(dsize)
        return train_loss, current_beta

    def test():
        test_loss = 0
        test_xy_iterator = test_loader_supervised.__iter__()
        test_x_iterator = test_loader_unsupervised_x.__iter__()
        test_y_iterator = test_loader_unsupervised_y.__iter__()
        bsize = test_loader_unsupervised_y.batch_size
        dsize = test_loader_unsupervised_y.dataset
        components = [0] * len(loss_components)
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

            x_up, x_down = x.to(device), torch.eye(vocabSize)[y[0].long()].to(device)
            x_up_split = x_u.to(device)
            x_down_split = torch.eye(vocabSize)[y_u[0].long()].to(device)
            loss = model.test(model_obj.model_args(
                x_down,
                x_up,
                x_down_split,
                x_up_split,
            ))
            test_loss += float(loss)
            for i, l in enumerate(model_obj.loss_components(
                x_down,
                x_up,
                x_down_split,
                x_up_split,
            )):
                components[i] += l
        for i, l in enumerate(components):
            loss_components[i].append(l * bsize / len(dsize))
        epoch_loss = test_loss * bsize / len(dsize)
        return epoch_loss


    train_losses, test_losses, coherences_truth, \
    coherences_i2s, coherences_s2i, coherences_joint = [], [], [], [], [], []
    coherences_i2s_valid, coherences_s2i_valid, coherences_joint_valid = \
        [], [], []

    loss_components = [[] for _ in range(10)]
    loss_components_train = [[] for _ in range(10)]

    for epoch in range(1, epochs+1):
        print('Model', model_obj.name, ', epoch', epoch, ', no labels', len(no_labels))
        loss, beta = train(epoch)
        train_losses.append(loss)
        betas.append(beta)
        t_l = test()
        coherence_truth, coherence_i2s, coherence_s2i, coherence_joint = \
            quantitative_eval(model_obj, test_loader_full)
        _, coherence_i2s_valid, coherence_s2i_valid, coherence_joint_valid = \
            quantitative_eval(model_obj, valid_loader_supervised)

        is_best = coherence_joint > best_loss
        best_loss = max(coherence_joint, best_loss)
        save_checkpoint(model_obj, is_best, '{}_{}_{}'.format(model_obj.name, len(no_labels), keyword))
        test_losses.append(t_l)
        qualitative_eval(model_obj, valid_loader_supervised, no_labels)

        coherences_truth.append(coherence_truth)

        coherences_i2s.append(coherence_i2s)
        coherences_s2i.append(coherence_s2i)
        coherences_joint.append(coherence_joint)

        coherences_i2s_valid.append(coherence_i2s_valid)
        coherences_s2i_valid.append(coherence_s2i_valid)
        coherences_joint_valid.append(coherence_joint_valid)

        result = pd.DataFrame({
            'train_loss': train_losses,
            'test_loss': test_losses,
            'n_parameters': [model_obj.get_number_of_parameters()] * len(train_losses),
            'coherence_truth': coherences_truth,
            'coherence_i2s': coherences_i2s,
            'coherence_s2i': coherences_s2i,
            'coherence_joint': coherences_joint,
            'coherence_i2s_valid': coherences_i2s_valid,
            'coherence_s2i_valid': coherences_s2i_valid,
            'coherence_joint_valid': coherences_joint_valid,
        })
        result.to_csv(f'{root}/results/{model_obj.name}_{len(no_labels)}_{keyword}.csv', index=False)

        loss_components_res = pd.DataFrame({i: l for i, l in enumerate(loss_components)})
        loss_components_res.to_csv(f'{root}/results/loss_components_{model_obj.name}_{len(no_labels)}_{keyword}.csv', index=False)

        loss_components_train_res = pd.DataFrame({i: l for i, l in enumerate(loss_components_train)})
        loss_components_train_res.to_csv(f'{root}/results/loss_components_train_{model_obj.name}_{len(no_labels)}_{keyword}.csv', index=False)

    result = pd.DataFrame({
        'train_loss': train_losses,
        'test_loss': test_losses,
        'n_parameters': [model_obj.get_number_of_parameters()]*len(train_losses),
        'coherence_truth': coherences_truth,
        'coherence_i2s': coherences_i2s,
        'coherence_s2i': coherences_s2i,
        'coherence_joint': coherences_joint,
        'coherence_i2s_valid': coherences_i2s_valid,
        'coherence_s2i_valid': coherences_s2i_valid,
        'coherence_joint_valid': coherences_joint_valid,
    })
    result.to_csv(f'{root}/results/{model_obj.name}_{len(no_labels)}_{keyword}.csv', index=False)


keyword = 'cub_captions'


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


# This is required because there are 10 captions per image.
# Allows easier reuse of the same image for the corresponding set of captions.
def resampler(dataset, idx):
    return idx // 10


tx = lambda data: torch.Tensor(data)
t_data = CUBSentences(root, split='train', transform=tx, max_sequence_length=maxSentLen)
s_data = CUBSentences(root, split='test', transform=tx, max_sequence_length=maxSentLen)

train_loader_sent = torch.utils.data.DataLoader(t_data, shuffle=False, **kwargs)
test_loader_sent = torch.utils.data.DataLoader(s_data, shuffle=False, **kwargs)

train_dataset = CUBImageFt(root, 'train', device)
test_dataset = CUBImageFt(root, 'test', device)
train_loader_images = torch.utils.data.DataLoader(train_dataset, shuffle=False, **kwargs)
test_loader_images = torch.utils.data.DataLoader(test_dataset, shuffle=False, **kwargs)

train_dataset._load_data()
test_dataset._load_data()


def pdist(sample_1, sample_2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances. Code
    adapted from the torch-two-sample library (added batching).
    You can find the original implementation of this function here:
    https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/util.py

    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(batch_size, n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(batch_size, n_2, d)``.
    norm : float
        The l_p norm to be used.
    batched : bool
        whether data is batched

    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (batch_size, n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    if len(sample_1.shape) == 2:
        sample_1, sample_2 = sample_1.unsqueeze(0), sample_2.unsqueeze(0)
    B, n_1, n_2 = sample_1.size(0), sample_1.size(1), sample_2.size(1)
    norms_1 = torch.sum(sample_1 ** 2, dim=-1, keepdim=True)
    norms_2 = torch.sum(sample_2 ** 2, dim=-1, keepdim=True)
    norms = (norms_1.expand(B, n_1, n_2)
             + norms_2.transpose(1, 2).expand(B, n_1, n_2))
    distances_squared = norms - 2 * sample_1.matmul(sample_2.transpose(1, 2))
    return torch.sqrt(eps + torch.abs(distances_squared)).squeeze()  # batch x K x latent


def NN_lookup(emb_h, emb, data):
    indices = pdist(emb.to(emb_h.device), emb_h).argmin(dim=0)
    # indices = torch.tensor(cosine_similarity(emb, emb_h.cpu().numpy()).argmax(0)).to(emb_h.device).squeeze()
    return data[indices]


def unproject(emb_h, search_split='train', te=train_dataset.ft_mat,
              td=train_dataset.data_mat, se=test_dataset.ft_mat, sd=test_dataset.data_mat):
    return NN_lookup(emb_h, te, td) if search_split == 'train' else NN_lookup(emb_h, se, sd)


t1, s1 = train_loader_images, test_loader_images
t2, s2 = train_loader_sent, test_loader_sent

N_data = 88550
no_labels_share = float(sys.argv[2])  # [0.0, 0.5, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999]
unsupervised_sampler_x, unsupervised_sampler_y, supervised_sampler = get_samplers(N_data, no_labels_share)

a, b = ResampleDataset(t1.dataset, resampler, size=len(t1.dataset) * 10), t2.dataset
print('Datasets')
print('size', len(t1.dataset))
print(a.size, len(b))


train_loader_full = torch.utils.data.DataLoader(TensorDataset([
    ResampleDataset(t1.dataset, resampler, size=len(t1.dataset) * 10),
    t2.dataset]), shuffle=True, **kwargs)
train_loader_supervised = torch.utils.data.DataLoader(TensorDataset([
    ResampleDataset(t1.dataset, resampler, size=len(t1.dataset) * 10),
    t2.dataset]), sampler=supervised_sampler, **kwargs)
train_loader_unsupervised_x = torch.utils.data.DataLoader(TensorDataset([
    ResampleDataset(t1.dataset, resampler, size=len(t1.dataset) * 10),
    t2.dataset]), sampler=unsupervised_sampler_x, **kwargs)
train_loader_unsupervised_y = torch.utils.data.DataLoader(TensorDataset([
    ResampleDataset(t1.dataset, resampler, size=len(t1.dataset) * 10),
    t2.dataset]), sampler=unsupervised_sampler_y, **kwargs)

N_test_data = int(29330 / 2)
unsupervised_sampler_x_test, unsupervised_sampler_y_test, supervised_sampler_test = get_samplers(N_test_data, no_labels_share)

test_loader_full = torch.utils.data.DataLoader(TensorDataset([
    ResampleDataset(s1.dataset, resampler, size=len(s1.dataset) * 10),
    s2.dataset]), shuffle=True, **kwargs)
test_loader_supervised = torch.utils.data.DataLoader(TensorDataset([
    ResampleDataset(s1.dataset, resampler, size=len(s1.dataset) * 10),
    s2.dataset]), sampler=supervised_sampler_test, **kwargs)
test_loader_unsupervised_x = torch.utils.data.DataLoader(TensorDataset([
    ResampleDataset(s1.dataset, resampler, size=len(s1.dataset) * 10),
    s2.dataset]), sampler=unsupervised_sampler_x_test, **kwargs)
test_loader_unsupervised_y = torch.utils.data.DataLoader(TensorDataset([
    ResampleDataset(s1.dataset, resampler, size=len(s1.dataset) * 10),
    s2.dataset]), sampler=unsupervised_sampler_y_test, **kwargs)

N_valid_data = int(29330 / 2)
unsupervised_sampler_x_valid, unsupervised_sampler_y_valid, supervised_sampler_valid = get_samplers(N_valid_data, 0, index_start=5000)
valid_loader_supervised = torch.utils.data.DataLoader(TensorDataset([
    ResampleDataset(s1.dataset, resampler, size=len(s1.dataset) * 10),
    s2.dataset]), sampler=supervised_sampler_valid, **kwargs)

eval_idx = [4965, 11376, 15755,  1507,  5709, 16578, 19913, 13336,  6487, 5196]


class SubsetSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


test_loader_supervised_eval = torch.utils.data.DataLoader(TensorDataset([
    ResampleDataset(s1.dataset, resampler, size=len(s1.dataset) * 10),
    s2.dataset]), sampler=SubsetSampler(eval_idx), **kwargs)

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

    return model_class(latentDim, {"lr": 1e-4}, q_x, q_y, p_x, p_y, q=q, q_star_y=q_star_y, q_star_x=q_star_x, y_coef=0.002)


model_obj = init_model(model_class)

no_labels_indices = set(np.random.choice(N, size=int(no_labels_share * N), replace=False))
if len(sys.argv) > 3 and sys.argv[3] == 'eval':
    is_best = not (len(sys.argv) > 4 and sys.argv[4] == 'current')
    name = '{}_{}_{}'.format(model_obj.name, len(no_labels_indices), keyword)
    load_checkpoint(model_obj, name, is_best=is_best)
    qualitative_eval(model_obj, test_loader_supervised_eval, no_labels_indices, keyword_run='paper')
    print('quant_results', quantitative_eval(model_obj, test_loader_supervised))
else:
    name = '{}_{}_{}'.format(model_obj.name, len(no_labels_indices), keyword)
    load_checkpoint(model_obj, name, is_best=False)
    run_semisupervised(model_obj, no_labels_indices)

