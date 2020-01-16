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

# Creating the parser to use parameters
parser = ap.ArgumentParser()
parser.add_argument('--folder', type=str, required=True, help='Folders to consider when creating the figures')
parser.add_argument('--baseline', type=str, required=True, help='Baseline to compare to')
parser.add_argument('--test', type=str, default='mnist', help='Name of the dataset that was used for training (for instance, mnist)')
parser.add_argument('--entropy_evaluation', type=str, default='notmnist', help='Name of the out-of-distribution dataset to apply the models on')
parser.add_argument('--n_nets', default=[10], nargs='+', required=False, help='Number of nets to use')
parser.add_argument('--save', action='store_true', help='Whether to save or show the image')

args = parser.parse_args()


def work(p, dirname, args):

    path_control = Path(args.baseline)

    with open(p / 'config.json', 'r') as fd:
        experiment_config = json.load(fd)

    # ------------------------------------------
    # Loading entropy stats for control
    # ------------------------------------------

    entropy_control_validation = h5py.File(path_control / 'stats/{}.h5'.format(args.test), 'r')
    entropy_control_evaluation = h5py.File(path_control / 'stats/{}.h5'.format(args.entropy_evaluation),
                                           'r')
    entro_validation_control_vals = entropy_control_validation.get('std')
    entro_evaluation_control_vals = entropy_control_evaluation.get('std')

    max_possible_entropy = tools.float_round(np.log(10), 2)  # entropy maximum if uniform distribution
    n_bins = 20

    # ------------------------------------------
    # Visualisation of the predictive entropy
    # ------------------------------------------
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 10), squeeze=False)

    # # read config
    with open(p / 'config.json', 'r') as fd:
        config = json.load(fd)

    # Read the entropy file
    path_stats = p / 'stats'
    entropy_subexp_validation = h5py.File(path_stats / '{}.h5'.format(args.test), 'r')
    entropy_subexp_evaluation = h5py.File(path_stats / '{}.h5'.format(args.entropy_evaluation), 'r')

    # --------------------------------------
    # First dataset: Validation or test dataset
    # In this setting we evaluate the generalization power of our method. We don't want to be too uncertain on those values.
    # --------------------------------------
    ax = axes[0, 0]
    ax.set_title('Entropy histogram for the\ntest dataset.')

    ax.set_xlim(-0.1, max_possible_entropy + 0.1)
    plt.setp(ax.get_xticklabels(), visible=True)

    for k in args.n_nets:
        entro_vals = entropy_subexp_validation.get('std')
        sns.distplot(entro_vals, hist=True, kde=False, bins=n_bins,
                     kde_kws={'shade': True, 'linewidth': 2},
                     hist_kws={"histtype": "stepfilled", 'range': (0, max_possible_entropy), 'linewidth': 2, 'log': False},
                     color='blue',
                     label='Repulsive ensemble', ax=ax)

    # Show control
    if args.baseline is not None:
        sns.distplot(entro_validation_control_vals, hist=True, kde=False, bins=n_bins,
                     kde_kws={'shade': True, 'linewidth': 2},
                     hist_kws={"histtype": "stepfilled", 'range': (0, max_possible_entropy), 'linewidth': 2, 'log': False},
                     color='orangered',
                     label='Deep ensemble', ax=ax)

    ax.set_xlabel('Entropy')
    ax.set_ylabel('Count')
    ax.legend()

    # --------------------------------------
    # Second dataset: Entropy evaluation dataset
    # --------------------------------------

    ax = axes[1, 0]

    # ax.set_title('lambda={}, bandwidth={}'.format(config['LAMBDA_REPULSIVE'], config['BANDWIDTH_REPULSIVE']))
    ax.set_title('Entropy histogram for an\nout of distribution dataset.')

    ax.set_xlim(-0.1, max_possible_entropy + 0.1)
    plt.setp(ax.get_xticklabels(), visible=True)

    for k in args.n_nets:
        entro_vals = entropy_subexp_evaluation.get('std')
        sns.distplot(entro_vals, hist=True, kde=False, bins=n_bins,
                     kde_kws={'shade': True, 'linewidth': 2},
                     hist_kws={"histtype": "stepfilled", 'range': (0, max_possible_entropy), 'linewidth': 2, 'log': False},
                     color='blue',
                     label='Repulsive ensemble', ax=ax)


    # Show control
    if args.baseline is not None:
        sns.distplot(entro_evaluation_control_vals, hist=True, kde=False, bins=n_bins,
                     kde_kws={'shade': True, 'linewidth': 2},
                     hist_kws={"histtype": "stepfilled", 'range': (0, max_possible_entropy), 'linewidth': 2, 'log': False},
                     color='orangered',
                     label='Deep ensemble', ax=ax)
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Count')
    ax.legend()

    # Now we save / show the plot
    plt.tight_layout()

    if args.save:
        path_savefig = p / 'entropy_{}_{}_lambda:{}_bandwidth:{}.pdf'.format(args.test, args.entropy_evaluation, config['lambda_repulsive'],
                                                                                     config['bandwidth_repulsive']
                                                                                     )
        fig.savefig(path_savefig)
    plt.close()


# ===============================================================================
# Going through the files and creating the figures
# ===============================================================================

# loop over all folders and if there is no std.h5 file we create it
folder = Path(args.folder)

for dirname in tqdm(os.listdir(folder)):
    if os.path.isdir(folder / dirname):
        work(folder / dirname, dirname, args)


