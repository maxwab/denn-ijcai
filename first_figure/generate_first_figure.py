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

lpconf = ['log/ensemble/config.json', 'log/last_ensemble/config.json','log/repulsive/config.json', 'log/last_repulsive/config.json']
lpmodels = ['log/ensemble/models', 'log/last_ensemble/models', 'log/repulsive/models', 'log/last_repulsive/models']
titles  = ['Ensemble', 'Ensemble', 'Repulsive', 'Repulsive']
lpos_txt = [0.0, 0.0, 0.0, 0.0]
dataset_seed = 2020

# Loading dataset
torch.manual_seed(dataset_seed)
np.random.seed(dataset_seed)
random.seed(dataset_seed)

x_train = (np.random.rand(10).reshape(-1, 1) - 1) / 2  # x between -0.5 and 0.0
y_train = f(x_train)

# Adding a single point at 0.35
nx = np.array([[.25]])
ny = f(nx)
X = np.concatenate([x_train, nx])
Y = np.concatenate([y_train, ny])

x_gt = np.linspace(-0.5, 0.5, 200).reshape(-1, 1)
y_gt = f(x_gt)

fig, axes = plt.subplots(2, 2, figsize=(10, 10), squeeze=False, sharex=True, sharey=True)

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

    ii, jj = np.unravel_index(i, (2, 2))
    ax = axes[ii, jj]
    ax.plot(x_gt, y_gt, 'k--', label='Ground truth')
    ax.fill_between(x.numpy().reshape(-1), m_ - 3 * s_, m_ + 3 * s_, color='b', alpha=.1)
    ax.fill_between(x.numpy().reshape(-1), m_ - 2 * s_, m_ + 2 * s_, color='b', alpha=.2)
    ax.fill_between(x.numpy().reshape(-1), m_ - s_, m_ + s_, color='b', alpha=.3, label='Standard deviations')
    ax.plot(x.numpy(), res[0, :].numpy(), c='m', label='Sample function')
    ax.scatter(x_train, y_train, marker='+', c='r', s=200, label='Data')
    if i in [1, 3]:
        blackswan = ax.scatter(0.25, f(0.25), marker='x', c='k', s=200, linewidth=3, label='Black swan event')
        if i == 1:
            ax.legend((blackswan,), ('Black swan event',))
    ax.axis([-.55, .55, -1.05, 1.05])
    if i == 0:
        ax.legend()
    if i == 0:
        ax.set_xlabel('(a)')
    elif i == 1:
        ax.set_xlabel('(b)')
    elif i == 2:
        ax.set_xlabel('(c)')
    elif i == 3:
        ax.set_xlabel('(d)')

fig.text(0, .35, 'DENN', rotation='vertical')
fig.text(0, .8, 'Ensemble', rotation='vertical')
fig.text(.3, .97, '$\mathcal{D}$')
fig.text(.6, .97, '$\mathcal{D}~\cup $ \{black swan event\}')
plt.tight_layout()
plt.subplots_adjust(hspace=.1, wspace=.05)

filename = 'illustration-objective'
path_figs = Path('img')
if not Path.exists(path_figs):
    os.makedirs(path_figs)
path_savefig = path_figs / '{}.pdf'.format(filename)
fig.savefig(path_savefig)
