import numpy as np
import torch
from dataset import RegressionDataset
from torch.utils.data import DataLoader


class repulsiveSampler(object):

    def __init__(self, x, var_noise=0.3, batch_size=20):
        idx_rand = np.random.choice(len(x), size=batch_size)
        xr = x[idx_rand].view(-1, 1)
        b = torch.distributions.Normal(loc=0.0, scale=var_noise)
        eps = b.sample((len(xr),)).view(-1, 1)
        self.x = xr + eps
        y = torch.zeros((len(self.x),))

        dataset = RegressionDataset(self.x, y)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.dataloader_iterator = iter(DataLoader(dataset, batch_size=batch_size, shuffle=True))

    def sample_batch(self):

        try:
            data, _ = next(self.dataloader_iterator)
        except StopIteration:
            self.dataloader_iterator = iter(self.dataloader)
            data, _ = next(self.dataloader_iterator)
        return data
