# ---------------------------
# Imports
from comet_ml import Experiment
import model
import torch, torch.optim as optim, torch.nn as nn
import random, json
import numpy as np
from tqdm import tqdm
import dataset
from functools import partial
import argparse as ap
from pathlib import Path
from src import tools
from torch.utils.data import DataLoader
import os
import sampler

from torchvision import transforms


def train(args, experiment=None, device=torch.device('cpu')):
    # ---------------------------------------
    # ---------------------------------------
    # Definition of the hyperaparameters ----
    # ---------------------------------------
    # ---------------------------------------

    # We create a configuration file with all the parameters
    namedataset = lambda x: x.split('/')[-1]
    nametrain = namedataset(args.train)
    if args.repulsive is not None:
        namerepulsive = namedataset(args.repulsive)
    else:
        namerepulsive = None
    model_name = 'train:{}_repulsive:{}_lambda:{}_bandwidth:{}'.format(nametrain, namerepulsive,
                                                                                          args.lambda_repulsive,
                                                                                          args.bandwidth_repulsive)

    if args.id is not None:
        model_name = model_name + '_{}'.format(args.id)

    try:
        # We create a configuration file with all the parameters
        savepath = Path(args.save_folder)
        if not Path.exists(savepath):
            os.makedirs(savepath)

        if not Path.exists(savepath / 'config.json'):  # Only create json if it does not exist
            with open(savepath / 'config.json', 'w') as fd:
                json.dump(vars(args), fd)
    except:
        pass
    experiment.log_parameters(vars(args))

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    # torch.set_num_threads(1)  # To comment out ?
    ACT_DIM = 2
    VAL_FREQ = 500
    EVAL_FREQ = 500

    net = model.PixelReacherPolicy(n_frames=3)
    net.to(device)

    torch.set_num_threads(1)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.3514, 0.3409, 0.3468], [0.0548, 0.0559, 0.0424])
    ])

    # ---------------------------
    # Load train loader

    train_dataset = dataset.ShapeColorDataset(root=args.train, split='train', transform=tfms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True)
    test_dataset = dataset.ShapeColorDataset(root=args.train, split='test', transform=tfms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False)

    if args.repulsive is not None:
        reference_net = model.PixelReacherPolicy(n_frames=3)
        reference_net.load_state_dict(torch.load(args.reference_net))
        reference_net.to(device)

        if args.repulsive != 'vaegan':
            # Load the repulsive sampler
            repulsive_dataset = dataset.ShapeColorDataset(root=args.repulsive,
                    split='test', transform=tfms, size=args.fixed_size_repulsive)
            repulsive_sampler = sampler.repulsiveSampler('regular', dataset=repulsive_dataset,
                                                         batch_size=args.batch_size_repulsive)

        else:
            repulsive_dataset = sampler.VAEGANDataset(noise_factor=args.noise_factor, size_memory=args.size_memory,
                                                      device='cuda:{}'.format(args.gpu))
            repulsive_sampler = sampler.repulsiveSampler('regular', dataset=repulsive_dataset,
                                                         batch_size=args.batch_size_repulsive)

        if args.evaluation is not None:
            # Load evaluation loader
            eval_dataset = dataset.ShapeColorDataset(root=args.evaluation, split='val', transform=tfms)
            eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size_test, shuffle=True)

    print('Finished loading the datasets.')

    # Partial functions
    if args.repulsive is not None:
        _optimize = partial(tools.optimize, bandwidth_repulsive=args.bandwidth_repulsive,
                            lambda_repulsive=args.lambda_repulsive)
    else:
        _optimize = tools.optimize

    # ---------------------------
    # Training
    step = 0

    for epoch in tqdm(range(args.n_epochs), desc='epochs'):
        # Training phase

        _tqdm = tqdm(train_loader, desc='batch')
        for j, batch_raw in enumerate(_tqdm):
            net.train()
            if args.repulsive is not None:
                br = repulsive_sampler.sample_batch()
                batch_repulsive = br.to(device)

            # optimization part # prepare the batch, we get images not vectors !
            x_raw, y = batch_raw
            batch = (x_raw.to(device), y.view(-1, ACT_DIM).to(device))

            if args.repulsive is not None:
                kwargs = {'reference_net': reference_net, 'batch_repulsive': batch_repulsive}
            else:
                kwargs = {}
            info_training = _optimize(net, optimizer, batch, add_repulsive_constraint=args.repulsive is not None,
                                      **kwargs)
            if args.verbose:
                _tqdm.set_description(
                    'Epoch {}/{}, loss: {:.4f}'.format(epoch + 1, args.n_epochs, info_training['loss']))

            # Log to Comet.ml
            for k, v in info_training.items():
                experiment.log_metric(k, float(v), step=step)
            step += 1

            if args.val_metrics:
                crit = nn.MSELoss()
                # ----------------------------------------------------
                # Validation phase
                if step % VAL_FREQ == 0:
                    # Evaluate on validation set
                    net.eval()
                    l = []
                    with torch.no_grad():
                        for j, batch_raw in enumerate(test_loader):
                            x_raw, y = batch_raw
                            x, y = x_raw.to(device), y.view(-1, ACT_DIM).to(device)
                            preds = net(x)
                            l.append(crit(preds, y).item())
                            if j >20:
                                break

                    # Compute statistics
                    print('Epoch {}/{}, val mse: {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(l)))
                    experiment.log_metric("val_mse", np.mean(l), step=step)

                if step % EVAL_FREQ == 0:
                    if args.evaluation is not None:
                        # Evaluate on validation set
                        net.eval()
                        l=[]
                        with torch.no_grad():
                            for j, batch_raw in enumerate(eval_loader):
                                x_raw, y = batch_raw
                                x, y = x_raw.to(device), y.view(-1, ACT_DIM).to(device)
                                preds = net(x)
                                l.append(crit(preds, y).item())
                                if j > 20:
                                    break
                        # Compute statistics
                        print('Epoch {}/{}, eval mse: {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(l)))
                        experiment.log_metric("eval_mse", np.mean(l), step=step)

    # ---------------------------------------
    # ---------------------------------------
    # Saving model

    if not Path.exists(savepath / 'models'):
        os.makedirs(savepath / 'models')

    model_path = savepath / 'models' / '{}_{}epochs.pt'.format(model_name, epoch + 1)
    if not Path.exists(model_path):
        torch.save(net.state_dict(), model_path)
    else:
        raise ValueError('Error trying to save file at location {}: File already exists'.format(model_path))


def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--train', required=True, type=str)
    parser.add_argument('--repulsive', type=str)
    parser.add_argument('--evaluation', type=str)
    parser.add_argument('--noise_factor', type=float, default=1.0)
    parser.add_argument('--size_memory', type=int, default=2048)
    parser.add_argument('--reference_net', type=str,
                        default='log/repulsive_default_color/reference_net_red_sphere.model')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--lambda_repulsive', type=float, default=0.0)
    parser.add_argument('--bandwidth_repulsive', type=float, default=0.0)
    parser.add_argument('--batch_size_train', type=int, default=64)
    parser.add_argument('--batch_size_test', type=int, default=64)
    parser.add_argument('--batch_size_repulsive', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--fixed_size_repulsive', type=int)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--id', type=int)
    parser.add_argument('--save_folder', type=str, default='log/reacher')
    parser.add_argument('--val_metrics', action='store_true')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--comet', action='store_true')

    args = parser.parse_args()

    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    # Log the experiment
    projname = 'pixel_reacher'
    experiment = Experiment(api_key="xxxx", project_name=projname, workspace="xxxx",
                            disabled=not args.comet)

    train(args, experiment, device)


if __name__ == '__main__':
    main()
