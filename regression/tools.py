import scipy.stats as st
import math
import torch
import numpy as np
import torch.nn as nn
from functools import partial


# Target function definition
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


# ANCHORING FUNCTIONS

def compute_norm_fac(net):
    """
    Computes the dictionary of factors to use per layer to ensure that each layer has an equal weight.
    :param net: Instance of a network.
    :return: Dict, the factors to use for each layer.
    """
    # Computation of the factor depending on the number of weights in the layer
    total_fac = {}
    for idx, (name, param) in enumerate(net.named_parameters()):
        n_params = np.prod(param.size())
        if name[-6:] == 'weight':
            norm_fac_per_weight = compute_norm_fac_per_weight(param.size()[1])
        total_fac[name] = 1 / (n_params * norm_fac_per_weight)  # Code not super pretty but this works anyway.
    return total_fac


def compute_norm_fac_per_weight(fan_in):
    """
    Computes the expectation of the weight initialization **squared** value.
    Weights are drawn from an uniform distribution.
    :param fan_in: Int, input size of the weight matrix.
    :return: Float, expectation of a squared initial weight.
    """
    ksi = 1. / math.sqrt(fan_in)
    return (1 / 3) * ksi ** 2


def criterion_anchoring_loss_full(current_params, init_params, fac_norm, batch_size):
    loss = torch.zeros(1)
    for idx, ((name_current, param_current), (name_init, param_init)) in enumerate(zip(current_params, init_params)):
        squared_diff = (param_current - param_init) ** 2
        a = torch.from_numpy(np.array(fac_norm[name_current] / batch_size)).type(torch.FloatTensor)
        b = torch.sum(squared_diff.type(torch.FloatTensor))

        loss += a * b
        # We divide by the batch_size because this loss is added for each batch!

    return loss


# REPULSIVE FUNCTION

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

