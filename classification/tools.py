import torch.nn as nn
import torch
from functools import partial
import losses
from math import ceil
import math
import numpy as np


def optimize(net, optimizer, batch, add_repulsive_constraint=False, **kwargs):
    criterion = nn.CrossEntropyLoss()

    if add_repulsive_constraint:
        criterion_repulsive = losses.CustomCrossEntropyLossContinuous(bandwidth=kwargs['bandwidth_repulsive'])

    info = {}
    x, y = batch  # x is an image and y is an integer !
    if 'beta' in kwargs.keys():
        output = net(x) + kwargs['beta'] * kwargs['prior'](x).detach()
    else:
        output = net(x)

    if not add_repulsive_constraint:
        loss = criterion(output, y)

        if 'lambda_anchoring' in kwargs.keys():
            anchoring_loss = criterion_anchoring_loss_full(net.named_parameters(), kwargs['prior'].named_parameters(), kwargs['fac_norm'],
                                                           batch[0].shape[0])

            loss += kwargs['lambda_anchoring'] * anchoring_loss[0]

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
        info['entropy_loss'] = entropy_loss.item()

        # total loss
        loss = data_loss + kwargs['lambda_repulsive'] * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # logging
    info['loss'] = loss.item()

    return info


def load_model(path, raw_model):
    try:
        if torch.cuda.is_available():
            raw_model.load_state_dict(torch.load(path, map_location='cuda:0'))
        else:
            raw_model.load_state_dict(torch.load(path, map_location='cpu'))
    except FileNotFoundError:
        print('Did not find the requested file at location {0}'.format(path))
        return -1
    except:
        print('Other error')
        import sys
        sys.exit(0)
    return raw_model


def float_round(num, places=0, direction=ceil):
    return direction(num * (10**places)) / float(10**places)


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
        if (name[-6:] == 'weight') and (name[-9:] != 'bn.weight'):
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
    loss = torch.zeros(1).to('cuda:0')
    for idx, ((name_current, param_current), (name_init, param_init)) in enumerate(zip(current_params, init_params)):
        squared_diff = (param_current - param_init) ** 2
        a = torch.from_numpy(np.array(fac_norm[name_current] / batch_size)).type(torch.FloatTensor).to('cuda:0')
        b = torch.sum(squared_diff.type(torch.FloatTensor)).to('cuda:0')

        loss += a * b
        # We divide by the batch_size because this loss is added for each batch!

    return loss
