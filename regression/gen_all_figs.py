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
from tools import f
import os
from copy import deepcopy

lpconf = ['log/bootstrapping/config.json', 'log/anchoring/config.json', 'log/repulsive/config.json']
lpmodels = ['log/bootstrapping/models', 'log/anchoring/models', 'log/repulsive/models']
titles  = ['Bootstrapping', 'Anchoring, $\lambda = 0.001$', 'Repulsive, $\lambda = 0.003$']
lpos_txt = [0.0, -.25, -.25]
dataset_seed = 2020
save = False

# Loading dataset
torch.manual_seed(dataset_seed)
np.random.seed(dataset_seed)
random.seed(dataset_seed)

x_train = (np.random.rand(10).reshape(-1, 1) - 1) / 2  # x between -0.5 and 0.0
y_train = f(x_train)

x_gt = np.linspace(-0.5, 0.5, 200).reshape(-1, 1)
y_gt = f(x_gt)

fig, axes = plt.subplots(2, 2, figsize=(10, 10), squeeze=False, sharex=True, sharey=True)

# DROPOUT
# ------------------------------------------------- 
pconf = 'log/dropout/config.json'
pmodel = 'log/dropout/models/dropout_lr:0.001_dr:0.200_5000epochs.pt'
title = 'Dropout, $p = 0.2$'

# Load configuration
with open(Path(pconf), 'r') as fd:
    config = json.load(fd)

# Load model
net = model.MLP(dropout_rate=config['dropout_rate'])
net.load_state_dict(torch.load(Path(pmodel)))

# Inference
out = []
x = torch.linspace(-.5, .5, 200).view(-1, 1)
for _ in range(50):
    net.eval()  # To keep the dropout
    with torch.no_grad():
        net.generate_mask()
        out.append(net(x).view(-1))

res = torch.stack(out, 0)
m_, s_ = res.mean(0), res.std(0)

ax = axes[0, 0]
ax.plot(x_gt, y_gt, 'k--', label='Ground truth')
ax.fill_between(x.numpy().reshape(-1), m_ - 3 * s_, m_ + 3 * s_, color='b', alpha=.1)
ax.fill_between(x.numpy().reshape(-1), m_ - 2 * s_, m_ + 2 * s_, color='b', alpha=.2)
ax.fill_between(x.numpy().reshape(-1), m_ - s_, m_ + s_, color='b', alpha=.3, label='Standard deviations')
ax.plot(x.numpy(), res[0, :].numpy(), c='m', label='Sample function')
ax.scatter(x_train, y_train, marker='+', c='r', s=200, label='Data')
ax.axis([-.55, .55, -1.05, 1.05])
ax.text(-0.1, 0.8, title, fontsize=24)
ax.legend(ncol=4, loc=(-.11,1), fontsize=17)

# OTHER
# ------------------------------------------------- 


for i, (pconf, pmodels, title, pt) in enumerate(zip(lpconf, lpmodels, titles, lpos_txt)):
    # Load configuration
    with open(Path(pconf), 'r') as fd:
        config = json.load(fd)

    # Load models
    net = model.MLP(dropout_rate=config['dropout_rate'])
    nets = []
    for modelname in [e for e in os.listdir(pmodels) if e[-2:] == 'pt']:
        mynet = deepcopy(net)
        mynet.load_state_dict(torch.load(Path(pmodels) / modelname))
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

    ii, jj = np.unravel_index(i+1, (2, 2))
    ax = axes[ii, jj]
    ax.plot(x_gt, y_gt, 'k--', label='Ground truth')
    ax.fill_between(x.numpy().reshape(-1), m_ - 3 * s_, m_ + 3 * s_, color='b', alpha=.1)
    ax.fill_between(x.numpy().reshape(-1), m_ - 2 * s_, m_ + 2 * s_, color='b', alpha=.2)
    ax.fill_between(x.numpy().reshape(-1), m_ - s_, m_ + s_, color='b', alpha=.3, label='Standard deviations')
    ax.plot(x.numpy(), res[0, :].numpy(), c='m', label='Sample function')
    ax.scatter(x_train, y_train, marker='+', c='r', s=200, label='Data')
    ax.axis([-.55, .55, -1.05, 1.05])
    ax.text(pt, 0.8, title, fontsize=24)
plt.tight_layout()
plt.subplots_adjust(hspace=.05, wspace=.05)
plt.show()


filename = '1d-regression'
path_figs = Path('img')
if not Path.exists(path_figs):
    os.makedirs(path_figs)
path_savefig = path_figs / '{}.pdf'.format(filename)
fig.savefig(path_savefig)
