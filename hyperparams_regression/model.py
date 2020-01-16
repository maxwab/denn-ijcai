import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.functional as F
from torch.distributions import Bernoulli


class MLP(nn.Module):

    def __init__(self, dropout_rate=0.0):
        """
        :param dropout_rate: Probability of a neuron being set to 0.
        """
        super().__init__()
        self.dropout_rate = dropout_rate

        self.fc1 = nn.Linear(1, 64)
        # self.fci_dropout = nn.Dropout(p=self.dropout_rate)
        self.fci_dropout = FixedDropout(p=self.dropout_rate)
        self.fc2 = nn.Linear(64, 64)
        # self.fc1_dropout = nn.Dropout(p=self.dropout_rate)
        self.fc1_dropout = FixedDropout(p=self.dropout_rate)
        self.fc3 = nn.Linear(64, 1)

        # Masks to use for dropout
        self.maski = None
        self.mask1 = None

    def forward(self, x):
        x = relu(self.fc1(x))
        x = self.fci_dropout(x, self.maski)
        x = relu(self.fc2(x))
        x = self.fc1_dropout(x, self.mask1)
        return self.fc3(x)

    def generate_mask(self, torch_seed=None):
        """
        Generate the dropout masks for stochastic inference.
        We use 1 - dropout_rate because dropout and bernoulli parameter p are 1 - the other.
        Fix torch_seed if we want to have a constant prediction
        :return:
        """
        self.mask1 = generate_mask(self.fc1, self.dropout_rate, torch_seed)
        self.mask2 = generate_mask(self.fc2, self.dropout_rate, torch_seed)

    def reset_mask(self):
        self.mask1 = None
        self.mask2 = None


class FixedDropout(_DropoutNd):
    r'''
    Applies dropout with a fixed given dropout mask.
    '''

    def forward(self, input, mask=None):
        if mask is not None:
            return input * mask
        else:
            return F.dropout(input, self.p, self.training, self.inplace)


def generate_mask(layer, dropout_rate, torch_seed=None):
    """
    Creates a fixed dropout mask to be used to sample a smooth function from a neural network.
    """
    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    mask = Bernoulli(
        torch.full_like(torch.FloatTensor(layer.out_features), 1.0 - dropout_rate)).sample() / (1.0 - dropout_rate)
    return mask
