import torch, torch.nn as nn
from functools import partial


def pairwise_rbf(y_ent_pts_new, y_ent_pts_old, std_pts):
    # computation of the weights
    return torch.mean(torch.exp(-(1 / (2 * std_pts**2)) * torch.norm(y_ent_pts_new - y_ent_pts_old, dim=1, keepdim=True)**2))


def optimize(net, optimizer, batch, add_repulsive_constraint=False, reverse_repulsive=False, **kwargs):
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
        info['entropy_loss'] = entropy_loss.item()

        # total loss
        loss = data_loss + kwargs['lambda_repulsive'] * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # logging
    info['loss'] = loss.item()

    return info


def sample_repulsive_batch(**kwargs):
    assert 'dataloader' in kwargs.keys(), 'dataset not specified.'
    dataloader_iterator = iter(kwargs['dataloader'])
    try:
        data, _ = next(dataloader_iterator)
    except StopIteration:
        dataloader_iterator = iter(kwargs['dataloader'])
        data, _ = next(dataloader_iterator)

    return data


class ReacherPolicyNN(nn.Module):

    def __init__(self, obs_dim, act_dim):
        super().__init__()

        # Architecture construction
        self.fc1 = nn.Linear(obs_dim, 120)
        self.fc2 = nn.Linear(120, 48)
        self.fc3 = nn.Linear(48, 20)
        self.fc4 = nn.Linear(20, act_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return self.fc4(x)
