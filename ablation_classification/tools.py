import torch.nn as nn
import torch
from functools import partial
import losses
from math import ceil


def optimize(net, optimizer, batch, add_repulsive_constraint=False, **kwargs):
    criterion = nn.CrossEntropyLoss()

    if add_repulsive_constraint:
        criterion_repulsive = losses.CustomCrossEntropyLossContinuous(bandwidth=kwargs['bandwidth_repulsive'])

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
