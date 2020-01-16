import h5py
import numpy as np
import argparse as ap
from pathlib import Path
import os
from tqdm import tqdm

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.9, rc={'text.usetex': True})
sns.set_style('whitegrid')

parser = ap.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='dataset NAME (not path)')
parser.add_argument('--save', action='store_true', help='Save or not the image, else show the figure')
parser.add_argument('--reverse_list', action='store_true', help='List documents in reverse order if true (for multiprocessing purposes) ')

args = parser.parse_args()

p1 = Path('log/train:red_sphere_repulsive:green_sphere_l:1_b:0.1')
p2 = Path('log/train:red_sphere_repulsive:green_sphere_l:10_b:0.1')
pb = Path('final_baseline_results/new_ensemble_train:red_sphere')

with h5py.File(Path(pb) / '{}.h5'.format(args.dataset), 'r') as cf:
    baseline_test_std = cf.get('std')[:]

with h5py.File(p1 / '{}.h5'.format(args.dataset), 'r') as f:
    test_std1 = f.get('std')[:]
with h5py.File(p2 / '{}.h5'.format(args.dataset), 'r') as f:
    test_std2 = f.get('std')[:]

# Preparing data for plots
# --> We take the mean since the differences between the actions do not seem to be significative
baseline_test = baseline_test_std.mean(1)
method_test1 = test_std1.mean(1)
method_test2 = test_std2.mean(1)

# display

# ===============================================================================
# showing the plots
# ===============================================================================
fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=False)

# Figure 1: on the test set (estimate the generalization)

plt.setp(ax.get_xticklabels(), visible=True)
bins = np.arange(0, 0.5, 0.01)  # defining the bins
# Proposed method
sns.distplot(method_test1, hist=True, kde=False, bins=bins,
             kde_kws={'shade': True, 'linewidth': 2},
             hist_kws={"histtype": "stepfilled", 'linewidth': 2, 'log': False},
             label=r'DENN, $\lambda=1$', ax=ax, color='blue')

# Proposed method
sns.distplot(method_test2, hist=True, kde=False, bins=bins,
             kde_kws={'shade': True, 'linewidth': 2},
             hist_kws={"histtype": "step", 'linewidth': 2, 'log': False},
             label=r'DENN, $\lambda=10$', ax=ax, color='blue')

# Baseline method
sns.distplot(baseline_test, hist=True, kde=False, bins=bins,
             kde_kws={'shade': True, 'linewidth': 2},
             hist_kws={"histtype": "stepfilled", 'linewidth': 2, 'log': False},
             label='Deep ensemble', ax=ax, color='orangered')

ax.set_xlabel('Mean std across actions')
ax.set_ylabel('Count')
ax.set_xlim(0.0, 0.5)
ax.legend(fontsize='small')
ax.set_title('Same color and shape')

plt.tight_layout()
if args.save:
    filename = 'entropy_dataset:{}.pdf'.format(args.dataset)
    plt.savefig(Path.cwd() / filename)
else:
    plt.show()
plt.close()
