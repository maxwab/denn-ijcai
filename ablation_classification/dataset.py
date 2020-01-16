import torchvision
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils import data
from torchvision.datasets.utils import download_url
from PIL import Image
import PIL
import numpy as np
import os
import errno
import h5py


class H5Dataset(Dataset):

    def __init__(self, path, transforms):
        self.h5f = h5py.File(path)
        self.tfms = transforms

    def __getitem__(self, index):
        data = self.h5f['sample'][index, :, :, :]
        target = self.h5f['target'][index]
        transformed_data = self.tfms(data)
        return transformed_data, target

    def __len__(self):
        return len(self.h5f['sample'])


class notMNIST(data.Dataset):
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
    test_file = 'test.pt'

    def __init__(self, root, train=False, transform=None, target_transform=None):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
            raise ValueError('No training set')
        else:
            self.test_data, self.test_labels = torch.load(self.root / self.processed_folder / self.test_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            raise ValueError('No training set')
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            raise ValueError('No training set')
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))


class KujuMNIST_DS(Dataset):
    urls = [
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz',
    ]
    base_filename = 'kmnist-{}-{}.npz'
    data_filepart = 'imgs'
    labels_filepart = 'labels'

    def __init__(self, folder, train_or_test='train', download=False, num_classes=10, max_items=None, tfms=None):
        self.root = os.path.expanduser(folder)
        if download:
            self.download()

        self.train = (train_or_test == 'train')

        self.data = np.load(os.path.join(self.root, self.base_filename.format(train_or_test, self.data_filepart)))
        self.data = self.data['arr_0']
        self.targets = np.load(os.path.join(self.root, self.base_filename.format(train_or_test, self.labels_filepart)))
        self.targets = self.targets['arr_0']
        self.c = num_classes
        self.max_items = max_items
        self.tfms = tfms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        cur_data = np.expand_dims(self.data[index], axis=-1)

        if self.tfms:
            cur_data = self.tfms(cur_data)

        target = int(self.targets[index])
        img, target = cur_data, target

        return img, target

    def __len__(self):
        if self.max_items:
            return self.max_items
        return len(self.data)

    def download(self):
        makedir_exist_ok(self.root)
        for url in self.urls:
            filename = url.rpartition('/')[-1]
            file_path = os.path.join(self.root, filename)
            download_url(url, root=self.root, filename=filename, md5=None)


class GenHelper(Dataset):
    # Code from https://gist.github.com/Fuchai/12f2321e6c8fa53058f5eb23aeddb6ab
    def __init__(self, mother, length, mapping):
        # here is a mapping from this index to the mother ds index
        self.mapping = mapping
        self.length = length
        self.mother = mother

    def __getitem__(self, index):
        return self.mother[self.mapping[index]]

    def __len__(self):
        return self.length


def train_valid_split(ds, split_fold=10, random_seed=None):
    '''
    # Code from https://gist.github.com/Fuchai/12f2321e6c8fa53058f5eb23aeddb6ab
    This is a pytorch generic function that takes a data.Dataset object and splits it to validation and training
    efficiently.

    :return:
    '''
    if random_seed != None:
        np.random.seed(random_seed)

    dslen = len(ds)
    indices = list(range(dslen))
    valid_size = dslen // split_fold
    np.random.shuffle(indices)
    train_mapping = indices[valid_size:]
    valid_mapping = indices[:valid_size]
    train = GenHelper(ds, dslen - valid_size, train_mapping)
    valid = GenHelper(ds, valid_size, valid_mapping)

    return train, valid, train_mapping, valid_mapping


def load_img_grayscale(path):
    res = Image.open(path)
    return PIL.ImageOps.grayscale(res)


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
