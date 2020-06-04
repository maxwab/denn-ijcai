from comet_ml import Experiment
import argparse as ap
import torch
import numpy as np
import random
from tools import f, optimize
import model
from dataset import RegressionDataset
from model import MLP
from tqdm import tqdm
import os
import json
from pathlib import Path
from functools import partial
from sampler import repulsiveSampler

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

parser = ap.ArgumentParser()
parser.add_argument('--type', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset_seed', type=int, default=2020)
parser.add_argument('--n_epochs', type=int, default=5000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--repulsive', type=str)
parser.add_argument('--lambda_repulsive', type=float, default=3e-3)
parser.add_argument('--batch_size_repulsive', type=int, default=20)
parser.add_argument('--dropout_rate', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--comet', action='store_true')
parser.add_argument('--save_folder', type=str, default='log/repulsive')
parser.add_argument('--id', type=int)

args = parser.parse_args()

# Logging

model_name = 'repulsive_lambda:{}'.format(args.lambda_repulsive)
if args.id is not None:
    model_name = model_name + '_{}'.format(args.id)

savepath = Path(args.save_folder)
try:
    if not Path.exists(savepath):
        os.makedirs(savepath)
except:
    pass

if not Path.exists(savepath / 'config.json'):  # Only create json if it does not exist
    with open(savepath / 'config.json', 'w') as fd:
        json.dump(vars(args), fd)

# Generate data and create dataset
torch.manual_seed(args.dataset_seed)
np.random.seed(args.dataset_seed)
random.seed(args.dataset_seed)

X = (np.random.rand(10).reshape(-1, 1) - 1) / 2  # x between -0.5 and 0.
Y = f(X)
X = torch.from_numpy(X).type(torch.FloatTensor)
Y = torch.from_numpy(Y).type(torch.FloatTensor)

# Adding a single point at 0.35
nx = torch.tensor([[.25]]).float()
ny = torch.from_numpy(f(nx)).type(torch.FloatTensor)
X = torch.cat([X, nx])
Y = torch.cat([Y, ny])

dataset = RegressionDataset(X, Y)

# Reproducibility
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

net = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

# Load reference net if defined
if args.repulsive is not None:
    reference_net = model.MLP(dropout_rate=args.dropout_rate)
    reference_net.load_state_dict(torch.load(Path(args.repulsive)))

# Update of the network parameters
train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Sampling a repulsive bandwidth parameter
alpha = -3
beta = -0.5
bandwidth_repulsive = float(10 ** (alpha + (beta - alpha) * np.random.rand()))

# Preparation of the optimization
if args.repulsive is not None:
    _optimize = partial(optimize, bandwidth_repulsive=bandwidth_repulsive, lambda_repulsive=args.lambda_repulsive)
else:
    _optimize = optimize

repulsive_sampler = repulsiveSampler(X, batch_size=args.batch_size_repulsive)

step = 0  # Number of batches seen
net.train()

# ----------------------------------------------------------------------
# Actual training
for epoch in tqdm(np.arange(args.n_epochs), disable=not args.verbose):

    for batch_idx, (data, target) in enumerate(train_loader):
        # Sample repulsive batch if required
        if args.repulsive is not None:
            br = repulsive_sampler.sample_batch()
            kwargs = {'reference_net': reference_net, 'batch_repulsive': br, 'bandwidth_repulsive': bandwidth_repulsive, 'lambda_repulsive': args.lambda_repulsive}
        else:
            kwargs = {}

        data, target = data.cpu(), target.cpu()
        info_batch = optimize(net, optimizer, batch=(data, target), add_repulsive_constraint=args.repulsive is not None,
                              **kwargs)
        step += 1

# Save the model
if not Path.exists(savepath / 'models'):
    os.makedirs(savepath / 'models')

model_path = savepath / 'models' / '{}_{}epochs.pt'.format(model_name, epoch + 1)
if not Path.exists(model_path):
    torch.save(net.state_dict(), model_path)
else:
    raise ValueError('Error trying to save file at location {}: File already exists'.format(model_path))
