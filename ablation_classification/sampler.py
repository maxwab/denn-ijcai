from torch.utils.data import Dataset
import torch
import math


class repulsiveSampler(object):

    def __init__(self, repulsive_dataset_type, **kwargs):
        self.type = repulsive_dataset_type
        self.batch_size = kwargs['batch_size']
        if 'noise_factor' in kwargs.keys():
            self.noise_factor = kwargs['noise_factor']
        else:
            self.noise_factor = 1.
        if self.type in ['FASHIONMNIST', 'NOISYMNIST', 'CIFAR100']:
            assert 'dataloader' in kwargs.keys(), 'dataset not specified.'
            self.dataloader = kwargs['dataloader']
            self.dataloader_iterator = iter(kwargs['dataloader'])
        elif self.type in ['UNIFORMNOISE', 'GAUSSIANNOISE']:
            pass
        else:
            raise ValueError('Problem, type {} not managed'.format(self.type))

    def sample_batch(self):
        if self.type in ['FASHIONMNIST', 'CIFAR100']:
            try:
                data, _ = next(self.dataloader_iterator)
            except StopIteration:
                self.dataloader_iterator = iter(self.dataloader)
                data, _ = next(self.dataloader_iterator)

        elif self.type == 'GAUSSIANNOISE':
            data = torch.randn(self.batch_size, 28, 28)

        elif self.type == 'UNIFORMNOISE':
            data = math.sqrt(3) * (2 * torch.rand(self.batch_size, 28, 28) - 1)

        elif self.type in ['NOISYMNIST']:
            try:
                data, _ = next(self.dataloader_iterator)
                data = data + self.noise_factor * torch.randn_like(data)  # Changes the variance

            except StopIteration:
                self.dataloader_iterator = iter(self.dataloader)
                data, _ = next(self.dataloader_iterator)
                data = data + self.noise_factor * torch.randn_like(data)

        return data
