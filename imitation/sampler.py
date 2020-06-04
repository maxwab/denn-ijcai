from pathlib import Path
import torch
import numpy as np
from skimage import color, transform
from torch.utils.data import DataLoader

class repulsiveSampler(object):
    input_channels = 3
    hidden_size = 64
    def __init__(self, repulsive_dataset_type, **kwargs):
        self.type = repulsive_dataset_type.lower()
        self.batch_size = kwargs['batch_size']
        if self.type in ['regular']:
            self.dataset = kwargs['dataset']
            self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
            self.dataloader_iterator = iter(self.dataloader)
        else:
            raise ValueError('Problem, type {} not managed'.format(self.type))
        self.ct = 0

    def sample_batch(self):

        if self.type in ['regular']:
            try:
                data, _ = next(self.dataloader_iterator)
            except StopIteration:
                self.ct += 1
                if self.ct == 10: # on ne reset qu'une fois pour éviter de trop souvent recalculer
                    self.dataset.reset() # new samples
                    self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
                    self.ct = 0
                self.dataloader_iterator = iter(self.dataloader)
                data, _ = next(self.dataloader_iterator)
            return data
        else:
            raise ValueError('Bad dataset type: {}'.format(self.type))



def transf(x):
    # in: x = (N, 3, 64, 64) torch.Tensor
    img = x.numpy() * 0.5 + 0.5
    npimg = np.transpose(img, (1, 2, 0))
    # Now we get a numpy image in color

    gray = color.rgb2gray(npimg)
    resized_gray = transform.resize(gray, (84, 84))

    final = (resized_gray - 0.4979) / 0.0348

    return final

def transf_color(x):

    # in: x = (N, 3, 64, 64) torch.Tensor
    img = x.numpy() * 0.5 + 0.5
    npimg = np.transpose(img, (1, 2, 0))
    # Now we get a numpy image in color

    resized_gray = transform.resize(npimg, (84, 84, 3))

    final = (resized_gray - 0.4979) / 0.0348
    final = np.transpose(final, (2, 0, 1))
    return final

from torch.utils import data

