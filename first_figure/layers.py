import torch
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.functional as F
from torch.distributions import Bernoulli


class FixedDropout(_DropoutNd):

    def forward(self, input, mask=None):
        if mask is not None:
            return input * mask
        else:
            return F.dropout(input, self.p, self.training, self.inplace)


def generate_mask(layer, dropout_rate, torch_seed=None):
    """
    Creates a fix dropout mask to be used to sample a smooth function from a neural network.
    :param layer: (nn.layer) Layer from pytorch.
    :param dropout_rate: (Float) Dropout rate: probability of setting to 0
    :return: Mask
    """
    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    mask = Bernoulli(
        torch.full_like(torch.FloatTensor(layer.out_features), 1.0 - dropout_rate)).sample() / (1.0 - dropout_rate)
    return mask
