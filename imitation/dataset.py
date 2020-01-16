from torch.utils.data import Dataset
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
import PIL
from torch.utils import data
import os
import h5py
import numpy as np

# Load premade datasets

class LeftDataset(Dataset):

    def __init__(self, train=True, path_data='datasets/left'):
        self.path_data = Path(path_data)
        self.train = train
        # Load h5 file
        if train:
            observes, actions, rewards, unscaled_obs = torch.load(self.path_data / 'training.pt')
        else:
            observes, actions, rewards, unscaled_obs = torch.load(self.path_data / 'test.pt')

        # now we want to transform the dataset to gray and rescaled images


    def __getitem__(self, index):
        data = self.observes[index].float()
        target = self.actions[index].float()
        return data, target

    def __len__(self):
        return self.observes.size(0)


class RightDataset(Dataset):

    def __init__(self, train=True, path_data='datasets/right'):
        self.path_data = Path(path_data)
        self.train = train
        # Load data
        if train:
            self.observes, self.actions, self.rewards, self.unscaled_obs = torch.load(self.path_data / 'training.pt')
        else:
            self.observes, self.actions, self.rewards, self.unscaled_obs = torch.load(self.path_data / 'test.pt')

    def __getitem__(self, index):
        data = self.observes[index].float()
        target = self.actions[index].float()
        return data, target

    def __len__(self):
        return self.observes.size(0)


class PixelDataset(data.Dataset):
    """`notMNIST `_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    processed_folder = 'processed'

    def __init__(self, root, train=False, transform=None, target_transform=None, pane='left'):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.train_file = 'px_trajs_ep:1000_pane:{}_seed:0.h5'.format(pane)
        self.test_file = 'px_trajs_ep:1000_pane:{}_seed:1.h5'.format(pane)

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
            # first we load the data
            with h5py.File(self.root / self.processed_folder / self.train_file, 'r') as h5f:
                self.observes = h5f.get('px_obs')[:]
                self.actions = h5f.get('actions')[:]
                self.rewards = h5f.get('rewards')[:]
                # then we transform it

        else:
            # first we load the data
            with h5py.File(self.root / self.processed_folder / self.test_file, 'r') as h5f:
                self.observes = h5f.get('px_obs')[:]
                self.actions = h5f.get('actions')[:]
                self.rewards = h5f.get('rewards')[:]
                # then we transform it

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        target = torch.from_numpy(self.actions[index]).float()
        img = Image.fromarray(self.observes[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.actions)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))


class SwimmerDataset(data.Dataset):
    """`notMNIST `_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    # mean : 0.4624
    # std : 0.4636
    processed_folder = 'processed'

    def __init__(self, root, train=False, transform=None, target_transform=None):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.test_file = 'px_trajs_swimmer_gray_ep:50_seed:0.h5'

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
           raise ValueError('No training set')

        else:
            # first we load the data
            with h5py.File(self.root / self.processed_folder / self.test_file, 'r') as h5f:
                self.observes = h5f.get('px_obs')[:]
                self.actions = h5f.get('actions')[:]
                self.rewards = h5f.get('rewards')[:]
                # then we transform it

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        target = torch.from_numpy(self.actions[index]).float()
        img = Image.fromarray(self.observes[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.actions)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))



class PixelColorDataset(data.Dataset):
    """`notMNIST `_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    # mean : [0.3514, 0.3408, 0.3468]
    # std : [0.0550, 0.0560, 0.0425]

    processed_folder = 'processed'


    def __init__(self, root, train=False, transform=None, target_transform=None, pane='left'):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.train_file = 'px_color_trajs_ep:1000_pane:{}_seed:0.h5'.format(pane)
        self.test_file = 'px_color_trajs_ep:1000_pane:{}_seed:0.h5'.format(pane)

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
            # first we load the data
            with h5py.File(self.root / self.processed_folder / self.train_file, 'r') as h5f:
                self.observes = h5f.get('px_obs')[:]
                #self.observes = self.observes.transpose((0, 2, 3, 1))
                self.actions = h5f.get('actions')[:]
                self.rewards = h5f.get('rewards')[:]
                # then we transform it

        else:
            # first we load the data
            with h5py.File(self.root / self.processed_folder / self.test_file, 'r') as h5f:
                self.observes = h5f.get('px_obs')[:]
                #self.observes = self.observes.transpose((0, 2, 3, 1))
                self.actions = h5f.get('actions')[:]
                self.rewards = h5f.get('rewards')[:]
                # then we transform it

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        target = torch.from_numpy(self.actions[index]).float()
        img = Image.fromarray(self.observes[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.actions)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))



#####################################################################################################


class ShapeColorDataset(data.Dataset):
    """`notMNIST `_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    processed_folder = 'processed'
    train_file = 'train.h5'
    test_file = 'test.h5'
    val_file = 'val.h5'

    def __init__(self, root, split='train', transform=None, target_transform=None, size=None):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set
        # validation is seed 2

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')
        
        if self.split == 'train':
            dataset_file = self.train_file
        elif self.split == 'test':
            dataset_file = self.test_file
        elif self.split == 'val':
            dataset_file = self.val_file
        else:
            raise ValueError('Bad split chosen: {}'.format(self.split))

        with h5py.File(self.root / self.processed_folder / dataset_file, 'r') as h5f:
            self.actions = h5f.get('actions')[:]
            self.observes = h5f.get('px_obs')[:]
            self.rewards = h5f.get('rewards')[:]

        if size is not None:
            idx = np.random.permutation(len(self))[:size]
            self.actions = self.actions[idx]
            self.observes  = self.observes[idx]
            self.rewards = self.rewards[idx]


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        target = torch.from_numpy(self.actions[index]).float()
        img = Image.fromarray(self.observes[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.actions)

    def reset(self):
        pass

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))
