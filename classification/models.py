import torch.nn as nn
import torch.nn.functional as F


class NNMNIST(nn.Module):

    def __init__(self, input_size, output_size, n_neurons=200):
        super().__init__()

        # Architecture construction
        self.fc1 = nn.Linear(input_size, n_neurons)
        self.fc1_bn = nn.BatchNorm1d(n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc2_bn = nn.BatchNorm1d(n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc3_bn = nn.BatchNorm1d(n_neurons)
        self.fco = nn.Linear(n_neurons, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc1_bn(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_bn(x)
        x = F.relu(self.fc3(x))
        x = self.fc3_bn(x)
        return self.fco(x)
