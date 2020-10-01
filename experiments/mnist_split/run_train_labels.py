from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from torchvision.models.resnet import ResNet, BasicBlock


class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3), bias=False)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, 100. * correct / len(test_loader.dataset)


class MNISTEvalDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_eval_loader(x_filepath, y_filepath):
    x = np.load(x_filepath)
    y = np.load(y_filepath).argmax(axis=1)
    eval_dataset = MNISTEvalDataset(x, y)
    return torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1000
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 2e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Evaluate against semi-supervised networks')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    root = "."
    filepath_saved_model = "mnist_cnn.pt"

    model = MnistResNet().to(device)
    if args.evaluate:
        model.load_state_dict(torch.load(filepath_saved_model))
        results = dict(
            model=[],
            n_unlabeled_batches=[],
            input=[],
            evaluation=[],
            test_loss=[],
            accuracy=[],
        )
        for model_name in ['SVAE', 'VAEVAE']:
            for n_batches in [570, 588, 594, 597, 599]:
                for input in ['full', 'up', 'down']:
                    for evaluation in ['best', 'current']:
                        filename_x = f'test_mnist_split/{evaluation}_{input}_mnist_split_{model_name}_{n_batches}.npy'
                        filename_y = f'test_mnist_split/{evaluation}_y_mnist_split_{model_name}_{n_batches}.npy'
                        eval_loader = make_eval_loader(filename_x, filename_y)
                        test_loss, accuracy = test(model, device, eval_loader)
                        results['model'].append(model_name)
                        results['n_unlabeled_batches'].append(n_batches)
                        results['input'].append(input)
                        results['evaluation'].append(evaluation)
                        results['test_loss'].append(test_loss)
                        results['accuracy'].append(accuracy)
        df_results = pd.DataFrame(results)
        df_results.to_csv('MNIST_split_evaluation_ResNet.csv', index=None)

    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                root=root,
                train=True,
                download=True,
                transform=transform,
            ),
            shuffle=True,
            batch_size=args.batch_size,
            **kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                root=root,
                train=False,
                transform=transform,
                download=True,
            ),
            shuffle=True,
            batch_size=args.test_batch_size,
            **kwargs
        )

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)

        if args.save_model:
            torch.save(model.state_dict(), filepath_saved_model)


if __name__ == '__main__':
    main()
