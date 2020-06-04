from comet_ml import Experiment
import argparse as ap
import torch
import numpy as np
import random
from tools import f
from dataset import RegressionDataset
from model import MLP, MLPbase
from tqdm import tqdm
from logger import JsonLogger

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
import os
import json

parser = ap.ArgumentParser()
parser.add_argument('--type', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset_seed', type=int, default=2020)
parser.add_argument('--n_epochs', type=int, default=5000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--dropout_rate', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--comet', action='store_true')
parser.add_argument('--save_folder', type=str, default='log/ensemble')
parser.add_argument('--id', type=int)

args = parser.parse_args()

# Logging
experiment = Experiment(api_key="XXX", project_name="final_regression", workspace="XXXX",
                        disabled=not args.comet)
experiment.log_parameters(vars(args))

model_name = 'ensemble'
if args.id is not None:
    model_name = model_name + '_{}'.format(args.id)

savepath = Path(args.save_folder)
try:
    if not Path.exists(savepath):
        os.makedirs(savepath)
except:
    pass

if not Path.exists(savepath / 'config.json'): # Only create json if it does not exist
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

torch.set_num_threads(1)

# Reproducibility
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

# Bootstrapping
dataset = RegressionDataset(X, Y)

net = MLP(args.dropout_rate)
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
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        step += 1

        experiment.log_metric('train_loss', loss.item(), step=step)

# Save the model
if not Path.exists(savepath / 'models'):
    os.makedirs(savepath / 'models')

model_path = savepath / 'models' / '{}_{}epochs.pt'.format(model_name, epoch+1)
if not Path.exists(model_path):
    torch.save(net.state_dict(), model_path)
else:
    raise ValueError('Error trying to save file at location {}: File already exists'.format(model_path))
