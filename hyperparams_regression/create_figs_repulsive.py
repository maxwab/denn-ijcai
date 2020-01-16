import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.9, rc={'text.usetex': True})
sns.set_style('whitegrid')
import argparse as ap
import model
from pathlib import Path
import json
import torch
import numpy as np
import random
import os
from copy import deepcopy


def f(x):
    return x


# Arguments
parser = ap.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to the configuration file of the experiment')
parser.add_argument('--models', type=str, required=True, help='Folder where the models are saved')
parser.add_argument('--dataset_seed', type=int, default=2020, help='For reproducibility')
parser.add_argument('--save', action='store_true', help='Where to save the image or show it')

args = parser.parse_args()

# Load configuration
with open(Path(args.config), 'r') as fd:
    config = json.load(fd)

# Load models
net = model.MLP(dropout_rate=config['dropout_rate'])
nets = []
for modelname in [e for e in os.listdir(args.models) if e[-2:] == 'pt']:
    mynet = deepcopy(net)
    mynet.load_state_dict(torch.load(Path(args.models) / modelname))
    nets.append(mynet)

# Inference
out = []
x = torch.linspace(-.5, .5, 200).view(-1, 1)
for net in nets:
    net.eval()  # To keep the dropout
    with torch.no_grad():
        out.append(net(x).view(-1))

res = torch.stack(out, 0)
m_, s_ = res.mean(0), res.std(0)

# Loading dataset
torch.manual_seed(args.dataset_seed)
np.random.seed(args.dataset_seed)
random.seed(args.dataset_seed)

X = np.linspace(-.5, .5, 13)
X = X[[0, 1, 2, 6, 7, 8, 9, 10, 11, 12]]  # Dropping some points
x_train = X.reshape(-1, 1)
y_train = f(x_train)

x_gt = np.linspace(-0.5, 0.5, 200).reshape(-1, 1)
y_gt = f(x_gt)

# Create figure
fig, axes = plt.subplots(1, 1, figsize=(5, 5), squeeze=False)
ax = axes[0, 0]
ax.plot(x_gt, y_gt, 'k--', label='Ground truth')
ax.fill_between(x.numpy().reshape(-1), m_ - 3 * s_, m_ + 3 * s_, color='b', alpha=.1)
ax.fill_between(x.numpy().reshape(-1), m_ - 2 * s_, m_ + 2 * s_, color='b', alpha=.2)
ax.fill_between(x.numpy().reshape(-1), m_ - s_, m_ + s_, color='b', alpha=.3, label='Standard deviations')
ax.plot(x.numpy(), res[0, :].numpy(), c='m', label='Sample function')
ax.scatter(x_train, y_train, marker='+', c='r', s=200, label='Training set')
ax.axis([-.55, .55, -1.05, 1.05])
ax.text(0.15, 0.8, 'DENN', fontsize=24)
ax.text(0.15, 0.65, '$\lambda = {}$'.format(config['lambda_repulsive']), fontsize=20)
# ax.legend()
plt.show()

if args.save:
    filename = 'repulsive_lambda-repulsive:{}'.format(config['lambda_repulsive'])
    path_figs = Path('img_final')
    if not Path.exists(path_figs):
        os.makedirs(path_figs)
    path_savefig = path_figs / '{}.pdf'.format(filename)
    fig.savefig(path_savefig)

    with open(path_figs / '{}.json'.format(filename), 'w') as fd:
        json.dump(vars(args), fd)
