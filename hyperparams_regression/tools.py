import scipy.stats as st
import math
import torch
import numpy as np
import torch.nn as nn
from functools import partial


# Target function definition
# ------------------------------------------------------------
def f(input_):
    r"""
    Bimodal function
    :param x:
    :return:
    """
    x = input_ + 0.5
    y_left = st.skewnorm(a=4, loc=.3, scale=.7).pdf(3 * x) / 1.6
    y_right = st.skewnorm(a=4, loc=.3, scale=.6).pdf(3 * (1 - x)) / 1.4
    return 2 * (y_left + y_right) - 1


# REPULSIVE FUNCTION
# ------------------------------------------------------------

def pairwise_rbf(y_ent_pts_new, y_ent_pts_old, std_pts):
    # computation of the weights
    return torch.mean(torch.exp(-(1 / (2 * std_pts**2)) * torch.norm(y_ent_pts_new - y_ent_pts_old, dim=1, keepdim=True)**2))


def optimize(net, optimizer, batch, add_repulsive_constraint=False, **kwargs):
    criterion = nn.MSELoss()
    if add_repulsive_constraint:
        criterion_repulsive = partial(pairwise_rbf, std_pts=kwargs['bandwidth_repulsive'])

    info = {}
    x, y = batch  # x is an image and y is an integer !
    output = net(x)

    if not add_repulsive_constraint:
        loss = criterion(output, y)
        info['data_loss'] = loss.item()
    else:
        data_loss = criterion(output, y)
        info['data_loss'] = data_loss.item()

        # entropy loss
        net.eval()
        y_rep = net(kwargs['batch_repulsive'])
        net.train()
        y_rep_ref = kwargs['reference_net'](kwargs['batch_repulsive']).detach()
        entropy_loss = criterion_repulsive(y_rep, y_rep_ref)  # close to 1 if the probs are the same, else close to 0
        info['repulsive_loss'] = entropy_loss.item()

        # total loss
        loss = data_loss + kwargs['lambda_repulsive'] * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # logging
    info['total_loss'] = loss.item()

    return info

