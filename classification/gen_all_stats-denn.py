import random
import numpy as np
import h5py
import argparse as ap
from pathlib import Path
import torch
import json
from tabulate import tabulate
from tqdm import tqdm
import tools
import stats
import models
from copy import deepcopy
import os
import torchvision
from torch.utils.data import DataLoader
import dataset

parser = ap.ArgumentParser()
parser.add_argument('--dataset', type=str, help='On which dataset to generate the statistics')
parser.add_argument('--final', action='store_true', help='Is this the final version (compute with the test sets and not the validation set)')
parser.add_argument('--id', type=int, help='id of the model')

args = parser.parse_args()

seed = 0
save = True
dataset_seed = 0
batch = 64
final = bool(args.final)
n_nets = 10

p = Path('log/train:mnist_repulsive:fashionmnist_l:0.05_b:10_seed:{}'.format(args.id))

# We also use this seed _directly_ to generate the training and validation set
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# ---------------------------------------------------------------------------------------------------------------
# Loading the datasets
# ---------------------------------------------------------------------------------------------------------------
# We load the training dset and the repulsive dset
if args.dataset.lower() == 'mnist':
    # Load transforms
    tfms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    if final:
        dset = torchvision.datasets.MNIST('../../../datasets/MNIST', train=False, download=True, transform=tfms)
    else:
        full_dset = torchvision.datasets.MNIST('../../../datasets/MNIST', train=True, download=True, transform=tfms)
        _, dset, _, _ = dataset.train_valid_split(full_dset, split_fold=10, random_seed=dataset_seed)
elif args.dataset.lower() == 'notmnist':
    tfms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.4240,), (0.4583,))])
    # Create the dset
    dset = dataset.notMNIST('../../../datasets/notMNIST', train=False, transform=tfms)
elif args.dataset.lower() == 'kmnist':
    tfms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1832,), (0.3405,))])
    # Create the dset
    dset = dataset.KujuMNIST_DS('../../../datasets/Kuzushiji-MNIST', train_or_test='test', download=True, tfms=tfms)
elif args.dataset.lower() == 'emnist':
    tfms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1733,), (0.3317,))])
    # Create the dset
    dset = torchvision.datasets.EMNIST('../../../datasets/emnist', split='letters', download=True, train=False, transform=tfms)
else:
    print('Bad dataset: can\'t load dataset {}'.format(args.dataset.lower()))

# Create the associated test loader
test_loader = DataLoader(dset, batch_size=batch, shuffle=False)

# ---------------------------------------------------------------------------------------------------------------
# Load the config file to retrieve the lambda and bandwidth parameter
# ---------------------------------------------------------------------------------------------------------------
with open(p / 'config.json', 'r') as fd:
    config = json.load(fd)
lambda_repulsive, bandwidth_repulsive = float(config['lambda_repulsive']), float(config['bandwidth_repulsive'])

path_stats = p / 'stats'
if not os.path.exists(path_stats):
    os.makedirs(path_stats)

# Loading the trained nets
if args.dataset.lower() in ['mnist', 'emnist', 'kmnist', 'notmnist']:
    base_model = models.NNMNIST(28 * 28, 10)
    reshape = lambda x: x.view(-1, 28 * 28)
else:
    raise ValueError('Can\'t load model for test dataset {}'.format(args.dataset.lower()))

all_trained_models = []
model_names = []

for model_name in [e for e in os.listdir(p / 'models') if e[-2:] == 'pt']:
    net = deepcopy(base_model)
    net = tools.load_model(p / 'models' / model_name, net)
    net.eval()
    all_trained_models.append(net)
    model_names.append(model_name)

nets = all_trained_models

# ---------------------------------------------------------------------------------------------------------------
# Validation accuracy on MNIST and entropy on notMNIST
# ---------------------------------------------------------------------------------------------------------------
print('Computing statistics for lambda = {} and bandwidth = {}'.format(lambda_repulsive, bandwidth_repulsive))

Z = []
for net in nets[:n_nets]:
    l, l_y = [], []
    for X, y in tqdm(test_loader):
        out = net(reshape(X).detach().cpu())
        l.append(out)
        l_y.append(y)
    Z.append(torch.cat(l, dim=0))
Y = torch.cat(l_y, dim=0)  # Computing the targets
Z = torch.stack(Z, dim=-1)
probs = torch.softmax(Z, 1)
entro_vals = stats.compute_entropy(probs)

if dataset == 'mnist':
    acc = stats.compute_accuracy(probs, Y)  # Compute accuracy by averaging the probas

# ---------------------------------------------------------------------------------------------------------------
# Generating statistics
# ---------------------------------------------------------------------------------------------------------------
if save:
    # Save the entropy values
    h5_filename = path_stats / '{}_final.h5'.format(args.dataset.lower())
    h5f = h5py.File(h5_filename, 'w')
    h5f.create_dataset('std', data=entro_vals)
    h5f.close()

    if dataset == 'mnist':
        # Save the accuracy values
        h5_filename = path_stats / 'acc_{}_final.h5'.format(args.dataset.lower())
        h5f = h5py.File(h5_filename, 'w')
        h5f.create_dataset('acc', data=acc)
        h5f.close()
