from comet_ml import Experiment
import argparse as ap
import torch
import numpy as np
import random
from tools import f, compute_norm_fac, criterion_anchoring_loss_full
from dataset import RegressionDataset
from model import MLP, MLPbase
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
import os
import json
from copy import deepcopy

parser = ap.ArgumentParser()
parser.add_argument('--type', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset_seed', type=int, default=2020)
parser.add_argument('--n_epochs', type=int, default=5000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--lambda_anchoring', type=float, default=1e-4)
parser.add_argument('--dropout_rate', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--comet', action='store_true')
parser.add_argument('--save_folder', type=str, default='log/anchoring')
parser.add_argument('--id', type=int)

args = parser.parse_args()

# Logging
experiment = Experiment(api_key="XXX", project_name="final_regression", workspace="XXXX",
                        disabled=not args.comet)
experiment.log_parameters(vars(args))

model_name = 'anchoring'
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
torch.set_num_threads(1)

X = (np.random.rand(10).reshape(-1, 1) - 1) / 2  # x between -0.5 and 0.
Y = f(X)
X = torch.from_numpy(X).type(torch.FloatTensor)
Y = torch.from_numpy(Y).type(torch.FloatTensor)

dataset = RegressionDataset(X, Y)

# Reproducibility
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

net = MLP(args.dropout_rate)
init_net = deepcopy(net)
fac_norm = compute_norm_fac(net)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

# Update of the network parameters
train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

step = 0  # Number of batches seen
net.train()

for epoch in tqdm(np.arange(args.n_epochs), disable=not args.verbose):
    experiment.log_current_epoch(epoch)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cpu(), target.cpu()

        optimizer.zero_grad()
        output = net(data)
        mse_loss = criterion(output, target)
        anchoring_loss = criterion_anchoring_loss_full(net.named_parameters(), init_net.named_parameters(), fac_norm,
                                                       args.batch_size)

        loss = mse_loss + args.lambda_anchoring * anchoring_loss
        loss.backward()
        optimizer.step()

        step += 1

        experiment.log_metric('train_loss', loss.item(), step=step)

# Save the model
if not Path.exists(savepath / 'models'):
    os.makedirs(savepath / 'models')

model_path = savepath / 'models' / '{}_{}epochs.pt'.format(model_name, epoch + 1)
if not Path.exists(model_path):
    torch.save(net.state_dict(), model_path)
else:
    raise ValueError('Error trying to save file at location {}: File already exists'.format(model_path))
