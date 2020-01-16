#!/usr/bin/env python
# coding: utf-8


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


args_dict = {
    'config': 'log/lv0.01/config.json',
    'models': 'log/lv0.01/models',
    'save': False,
    'dataset_seed': 2020
}
args = Bunch(args_dict)

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

from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.gaussian_process.kernels import RBF, WhiteKernel as WK, ConstantKernel as C


def f(x):
    return x


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

# ------------------------------------------------------------
# Train the GP
x = np.linspace(-.5, .5, 100).reshape(-1, 1)

# Kernel values

# Flat:
Cval = .05
Bval = .5
alphaval = 1e-5

kernel = C(Cval, (Cval, Cval)) * RBF(Bval, (Bval, Bval))
gp = GP(kernel=kernel, n_restarts_optimizer=9, alpha=alphaval)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(x_train, y_train.reshape(-1))

# Make the prediction on the meshed x-axis (ask for MSE as well)
m_, s_ = gp.predict(x, return_std=True)
m_ = m_.reshape(-1)

# Sample function
y_samples = gp.sample_y(x, 10)

# Create figure
fig, axes = plt.subplots(1, 1, figsize=(5, 5), squeeze=False)
ax = axes[0, 0]
ax.plot(x_gt, y_gt, 'k--', label='Ground truth')
ax.fill_between(x.reshape(-1), m_ - 3 * s_, m_ + 3 * s_, color='b', alpha=.1)
ax.fill_between(x.reshape(-1), m_ - 2 * s_, m_ + 2 * s_, color='b', alpha=.2)
ax.fill_between(x.reshape(-1), m_ - s_, m_ + s_, color='b', alpha=.3, label='Standard deviations')
ax.plot(x, y_samples[:, 0], c='m')
ax.scatter(x_train, y_train, marker='+', c='r', s=200, label='Training set')
ax.axis([-.55, .55, -1.05, 1.05])
ax.text(-.15, 0.8, 'Gaussian process', fontsize=24)
ax.text(-.45, 0.65, '$C={}, \sigma={}, \epsilon={}$'.format(Cval, Bval, alphaval), fontsize=20)
# ax.legend()
plt.show()

filename = 'gp_c-5e-2_b5e-1_a1e-5'
path_figs = Path('img_final')
if not Path.exists(path_figs):
    os.makedirs(path_figs)
path_savefig = path_figs / '{}.pdf'.format(filename)
fig.savefig(path_savefig)

with open(path_figs / '{}.json'.format(filename), 'w') as fd:
    json.dump(vars(args), fd)


# In[6]:


# Train the GP
x = np.linspace(-.5, .5, 100).reshape(-1, 1)

# Kernel values

# 0.0001:
Cval = .05
Bval = .2
alphaval = 1e-5

kernel = C(Cval, (Cval, Cval)) * RBF(Bval, (Bval, Bval))
gp = GP(kernel=kernel, n_restarts_optimizer=9, alpha=alphaval)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(x_train, y_train.reshape(-1))

# Make the prediction on the meshed x-axis (ask for MSE as well)
m_, s_ = gp.predict(x, return_std=True)
m_ = m_.reshape(-1)

# Sample function
y_samples = gp.sample_y(x, 10)

# Create figure
fig, axes = plt.subplots(1, 1, figsize=(5, 5), squeeze=False)
ax = axes[0, 0]
ax.plot(x_gt, y_gt, 'k--', label='Ground truth')
ax.fill_between(x.reshape(-1), m_ - 3 * s_, m_ + 3 * s_, color='b', alpha=.1)
ax.fill_between(x.reshape(-1), m_ - 2 * s_, m_ + 2 * s_, color='b', alpha=.2)
ax.fill_between(x.reshape(-1), m_ - s_, m_ + s_, color='b', alpha=.3, label='Standard deviations')
ax.plot(x, y_samples[:, 0], c='m')
ax.scatter(x_train, y_train, marker='+', c='r', s=200, label='Training set')
ax.axis([-.55, .55, -1.05, 1.05])
ax.text(-.15, 0.8, 'Gaussian process', fontsize=24)
ax.text(-.45, 0.65, '$C={}, \sigma={}, \epsilon={}$'.format(Cval, Bval, alphaval), fontsize=20)
plt.show()


filename = 'gp_c-5e-2_b2e-1_a1e-5'
path_figs = Path('img_final')
if not Path.exists(path_figs):
    os.makedirs(path_figs)
path_savefig = path_figs / '{}.pdf'.format(filename)
fig.savefig(path_savefig)

with open(path_figs / '{}.json'.format(filename), 'w') as fd:
    json.dump(vars(args), fd)

# Train the GP
x = np.linspace(-.5, .5, 100).reshape(-1, 1)

# Kernel values

# 0.001:
Cval = .05
Bval = .14
alphaval = 1e-5


# Script

kernel = C(Cval, (Cval, Cval)) * RBF(Bval, (Bval, Bval))
gp = GP(kernel=kernel, n_restarts_optimizer=9, alpha=alphaval)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(x_train, y_train.reshape(-1))

# Make the prediction on the meshed x-axis (ask for MSE as well)
m_, s_ = gp.predict(x, return_std=True)
m_ = m_.reshape(-1)

# Sample function
y_samples = gp.sample_y(x, 10)

# Create figure
fig, axes = plt.subplots(1, 1, figsize=(5, 5), squeeze=False)
ax = axes[0, 0]
ax.plot(x_gt, y_gt, 'k--', label='Ground truth')
ax.fill_between(x.reshape(-1), m_ - 3 * s_, m_ + 3 * s_, color='b', alpha=.1)
ax.fill_between(x.reshape(-1), m_ - 2 * s_, m_ + 2 * s_, color='b', alpha=.2)
ax.fill_between(x.reshape(-1), m_ - s_, m_ + s_, color='b', alpha=.3, label='Standard deviations')
ax.plot(x, y_samples[:, 0], c='m')
ax.scatter(x_train, y_train, marker='+', c='r', s=200, label='Training set')
ax.axis([-.55, .55, -1.05, 1.05])
ax.text(-.15, 0.8, 'Gaussian process', fontsize=24)
ax.text(-.45, 0.65, '$C={}, \sigma={}, \epsilon={}$'.format(Cval, Bval, alphaval), fontsize=20)
# ax.legend()
plt.show()

filename = 'gp_c-5e-2_b14e-2_a1e-5'
path_figs = Path('img_final')
if not Path.exists(path_figs):
    os.makedirs(path_figs)
path_savefig = path_figs / '{}.pdf'.format(filename)
fig.savefig(path_savefig)

with open(path_figs / '{}.json'.format(filename), 'w') as fd:
    json.dump(vars(args), fd)

# Train the GP
x = np.linspace(-.5, .5, 100).reshape(-1, 1)

# Kernel values

# 0.01:
Cval = .05
Bval = .08
alphaval = 1e-3

# Script

kernel = C(Cval, (Cval, Cval)) * RBF(Bval, (Bval, Bval))
gp = GP(kernel=kernel, n_restarts_optimizer=9, alpha=alphaval)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(x_train, y_train.reshape(-1))

# Make the prediction on the meshed x-axis (ask for MSE as well)
m_, s_ = gp.predict(x, return_std=True)
m_ = m_.reshape(-1)

# Sample function
y_samples = gp.sample_y(x, 10)

# Create figure
fig, axes = plt.subplots(1, 1, figsize=(5, 5), squeeze=False)
ax = axes[0, 0]
ax.plot(x_gt, y_gt, 'k--', label='Ground truth')
ax.fill_between(x.reshape(-1), m_ - 3 * s_, m_ + 3 * s_, color='b', alpha=.1)
ax.fill_between(x.reshape(-1), m_ - 2 * s_, m_ + 2 * s_, color='b', alpha=.2)
ax.fill_between(x.reshape(-1), m_ - s_, m_ + s_, color='b', alpha=.3, label='Standard deviations')
ax.plot(x, y_samples[:, 0], c='m')
ax.scatter(x_train, y_train, marker='+', c='r', s=200, label='Training set')
ax.axis([-.55, .55, -1.05, 1.05])
ax.text(-.15, 0.8, 'Gaussian process', fontsize=24)
ax.text(-.45, 0.65, '$C={}, \sigma={}, \epsilon={}$'.format(Cval, Bval, alphaval), fontsize=20)
plt.show()


filename = 'gp_c-5e-2_b8e-2_a1e-3'
path_figs = Path('img_final')
if not Path.exists(path_figs):
    os.makedirs(path_figs)
path_savefig = path_figs / '{}.pdf'.format(filename)
fig.savefig(path_savefig)

with open(path_figs / '{}.json'.format(filename), 'w') as fd:
    json.dump(vars(args), fd)
