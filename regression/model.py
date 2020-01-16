import torch.nn as nn
from layers import FixedDropout, generate_mask
from torch.nn.functional import relu

class MLP(nn.Module):

    def __init__(self, dropout_rate=0.1):
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

class MLPbase(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(-1, 1)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def name(self):
        return 'MLP'