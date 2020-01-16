from torch.utils.data import Dataset
import torch
import torch.nn.functional as F


class RegressionDataset(Dataset):
    def __init__(self, x, y):
        # self.x = torch.from_numpy(x).view(-1, 1).type(torch.FloatTensor)
        # self.y = torch.from_numpy(y).view(-1).type(torch.FloatTensor)
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        # Retrieve the item
        sample = self.x[item]
        target = self.y[item]
        return sample, target
