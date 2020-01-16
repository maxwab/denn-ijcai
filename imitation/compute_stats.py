import random
import numpy as np
import argparse as ap
from pathlib import Path
import h5py
import torch
import json
from copy import deepcopy
from tqdm import tqdm
import model

# First step: we read the arguments
import os
from src import tools
from torch.utils.data import DataLoader
import torch.nn as nn
import dataset
from torchvision import transforms

parser = ap.ArgumentParser()
parser.add_argument('--folder', type=str, required=True, help='compute statistics for folders in this directory')
parser.add_argument('--prefix', type=str, help='prefix to use for folders', required=True)
parser.add_argument('--dataset', type=str, help='path to dataset')
parser.add_argument('--n_nets', type=int, default=10, help='max number of nets to use')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch', type=int, default=512, help='samples per batch')
parser.add_argument('--device', type=str, default='cuda:0', help='which device to use')
parser.add_argument('--reverse_list', action='store_true', help='List documents in reverse order if true (for multiprocessing purposes) ')
parser.add_argument('--final', action='store_true')

args = parser.parse_args()

OBS_DIM = 12
ACT_DIM = 2
device = torch.device(args.device)
if args.seed is not None:
    # We also use this seed _directly_ to generate the training and validation set
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

# Same normalization for all
tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.3514, 0.3409, 0.3468], [0.0548, 0.0559, 0.0424])
])

split = 'test' if args.final else 'val'

dataset = dataset.ShapeColorDataset(root=args.dataset, split=split, transform=tfms)
data_loader = DataLoader(dataset, batch_size=args.batch, shuffle=False)
raw_net = model.PixelReacherPolicy(n_frames=3)


def work(path_experiment, nets, dataloader, dataset_name):
    # For each net we compute the probabilities and then we extract the metrics.
    with torch.no_grad():
        mse = nn.MSELoss()
        Z = []
        for net in nets[:args.n_nets]:
            l, l_y = [], []
            lim=0
            for X, y in tqdm(dataloader):
                X, y = X.to(device), y.to(device)
                out = net(X).detach()
                l.append(out)
                l_y.append(y)
                lim += 1
                if lim > n_mini_batch_before_break:
                    break
            Z.append(torch.cat(l, dim=0))
        ood_targets = torch.cat(l_y, dim=0)
        ood_preds = torch.stack(Z, dim=-1)
        values_std = ood_preds.std(-1).cpu()  # The measure of uncertainty is the prediction variance

    # -------------------------------------------
# Evaluation on an OOD dataset: are our predictions worse? what is the standard deviation on it?
    with h5py.File(path_experiment / '{}.h5'.format(dataset_name), 'a') as h5f:
        h5f.create_dataset('std', data=values_std.detach().numpy())


# ================================================================================
# Beginning of the script
# ================================================================================

# loop over all folders and if there is no std.h5 file we create it
folder = Path(args.folder)

for dirname in tqdm(os.listdir(folder)):
    if os.path.isdir(folder / dirname):
        if len(os.listdir(folder / dirname / 'models')) > 0:  # Ensure that there is at least 1 model
            path_experiment = folder / dirname
            print('Working on folder {} ...'.format(dirname))

            # Loading the trained nets
            nets = []
            end_name = '10epochs.pt'
            for net_name in [e for e in os.listdir(path_experiment / 'models') if e[-len(end_name):] == end_name]:
                net = deepcopy(raw_net)
                net.load_state_dict(torch.load(path_experiment / 'models' / net_name, map_location=args.device))
                net.eval()
                net = nn.DataParallel(net)
                net = net.to(device)
                nets.append(net.to(device))

            # Creating data
            dataset_name = args.dataset.split('/')[-1]
            if args.final:
                dataset_name += '_final'
            if '{}.h5'.format(dataset_name) not in [e for e in os.listdir(folder / dirname)]:
                work(folder / dirname, nets, data_loader, dataset_name)
