import h5py
import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.9, rc={'text.usetex': True})
sns.set_style('whitegrid')

from tqdm import tqdm
import argparse as ap
import numpy as np
import tools
import json
from pathlib import Path
import os

p = Path('cv/train:mnist_repulsive:uniformnoise_l:50._b:10.')
dirname = 'train:mnist_repulsive:uniformnoise_l:50._b:10.'
path_control = Path('log/control')
eval_datasets = ['mnist_final', 'notmnist_final', 'kmnist_final']
titles = ['MNIST', 'notMNIST', 'KMNIST']
maxval = 15000
label = 'DENN'
save = False

max_possible_entropy = tools.float_round(np.log(10), 2)  # entropy maximum if uniform distribution
n_bins = 20

# # read config
with open(p / 'config.json', 'r') as fd:
    config = json.load(fd)

# Read the entropy file
path_stats = p / 'stats'

# ------------------------------------------
# Visualisation of the predictive entropy
# ------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(10, 5), squeeze=False, sharey=True)

# ------------------------------------------
# Loading entropy stats for control
# ------------------------------------------
for i, (eval_dataset, title) in enumerate(zip(eval_datasets, titles)):

    ax = axes[0, i]
    entropy_control_validation = h5py.File(path_control / 'stats/{}.h5'.format(eval_dataset), 'r')
    entro_validation_control_vals = entropy_control_validation.get('std')

    entropy_subexp_validation = h5py.File(path_stats / '{}.h5'.format(eval_dataset), 'r')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # --------------------------------------
    # First dataset: Validation or test dataset
    # In this setting we evaluate the generalization power of our method. We don't want to be too uncertain on those values.
    # --------------------------------------
    # plt.title('Entropy histogram \non MNIST')
    ax.set_xlim(-0.1, max_possible_entropy + 0.1)
    ax.set_ylim(-0.1, maxval)
    # plt.setp(ax.get_xticklabels(), visible=True)

    entro_vals = entropy_subexp_validation.get('std')
    sns.distplot(entro_vals, hist=True, kde=False, bins=n_bins,
                 kde_kws={'shade': True, 'linewidth': 2},
                 hist_kws={"histtype": "stepfilled", 'range': (0, max_possible_entropy), 'linewidth': 2, 'log': False},
                 color='blue',
                 label=label, ax=ax)

    sns.distplot(entro_validation_control_vals, hist=True, kde=False, bins=n_bins,
                 kde_kws={'shade': True, 'linewidth': 2},
                 hist_kws={"histtype": "stepfilled", 'range': (0, max_possible_entropy), 'linewidth': 2, 'log': False},
                 color='orangered',
                 label='Deep ensemble', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predictive entropy')
    if i == 2:
        ax.legend()
    if i == 0:

        ax.set_ylabel('Count')
    else:
        ax.tick_params(axis=u'both', which=u'both', length=0)
        ax.set_ylabel('')


# Now we save / show the plot
# ax.set_aspect('equal')
fig.tight_layout()
plt.subplots_adjust(wspace=.1)

# Now we save the figure
figname = 'mnist_notmnist-uniform.pdf'
p_fig = Path('img') / figname
if not Path.exists(Path('img')):
    os.makedirs(Path('img'))
fig.savefig(p_fig)


p = Path('cv/train:mnist_repulsive:gaussiannoise_l:100._b:1.')
dirname = 'train:mnist_repulsive:gaussiannoise_l:100._b:1.'
path_control = Path('log/control')
eval_datasets = ['mnist_final', 'notmnist_final', 'kmnist_final']
titles = ['MNIST', 'notMNIST', 'KMNIST']
maxval = 15000
label = 'DENN'
save = False

max_possible_entropy = tools.float_round(np.log(10), 2)  # entropy maximum if uniform distribution
n_bins = 20

# # read config
with open(p / 'config.json', 'r') as fd:
    config = json.load(fd)

# Read the entropy file
path_stats = p / 'stats'

# ------------------------------------------
# Visualisation of the predictive entropy
# ------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(10, 5), squeeze=False, sharey=True)

# ------------------------------------------
# Loading entropy stats for control
# ------------------------------------------
for i, (eval_dataset, title) in enumerate(zip(eval_datasets, titles)):

    ax = axes[0, i]

    entropy_control_validation = h5py.File(path_control / 'stats/{}.h5'.format(eval_dataset), 'r')
    entro_validation_control_vals = entropy_control_validation.get('std')

    entropy_subexp_validation = h5py.File(path_stats / '{}.h5'.format(eval_dataset), 'r')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # --------------------------------------
    # First dataset: Validation or test dataset
    # In this setting we evaluate the generalization power of our method. We don't want to be too uncertain on those values.
    # --------------------------------------
    # plt.title('Entropy histogram \non MNIST')
    ax.set_xlim(-0.1, max_possible_entropy + 0.1)
    ax.set_ylim(-0.1, maxval)
    # plt.setp(ax.get_xticklabels(), visible=True)

    entro_vals = entropy_subexp_validation.get('std')
    sns.distplot(entro_vals, hist=True, kde=False, bins=n_bins,
                 kde_kws={'shade': True, 'linewidth': 2},
                 hist_kws={"histtype": "stepfilled", 'range': (0, max_possible_entropy), 'linewidth': 2, 'log': False},
                 color='blue',
                 label=label, ax=ax)

    sns.distplot(entro_validation_control_vals, hist=True, kde=False, bins=n_bins,
                 kde_kws={'shade': True, 'linewidth': 2},
                 hist_kws={"histtype": "stepfilled", 'range': (0, max_possible_entropy), 'linewidth': 2, 'log': False},
                 color='orangered',
                 label='Deep ensemble', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predictive entropy')
    if i == 2:
        ax.legend()
    if i == 0:

        ax.set_ylabel('Count')
    else:
        ax.tick_params(axis=u'both', which=u'both', length=0)
        ax.set_ylabel('')

# Now we save / show the plot
# ax.set_aspect('equal')
fig.tight_layout()
plt.subplots_adjust(wspace=.1)

# Now we save the figure
figname = 'mnist_notmnist-gaussian.pdf'
p_fig = Path('img') / figname
if not Path.exists(Path('img')):
    os.makedirs(Path('img'))
fig.savefig(p_fig)
