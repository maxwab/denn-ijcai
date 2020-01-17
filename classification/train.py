from comet_ml import Experiment
import torch, torch.optim as optim, torch.nn as nn

import models
import tools
import sampler
import random, json
import numpy as np
from tqdm import tqdm
import dataset
import torchvision
from functools import partial
import argparse as ap
from pathlib import Path
import os


def train(args, experiment=None, device=None):
    # ---------------------------------------
    # Definition of the hyperaparameters
    # ---------------------------------------

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Loading dataset parameters
    if args.train.lower() == 'mnist':
        net = models.NNMNIST(28 * 28, 10).to(device)
        if args.beta > 0.0:
            prior = models.NNMNIST(28 * 28, 10).to(device)
            prior.eval()
        else:
            prior = None
        # Load transforms
        tfms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        full_dset = torchvision.datasets.MNIST('../../../datasets/MNIST', train=True, download=False, transform=tfms)
        prepr = lambda x: x.view(-1, 28 * 28)
    else:
        raise ValueError('Bad training dataset selected: {}'.format(args.train.lower()))

    # Create training and validation split
    train_dset, val_dset, _, _ = dataset.train_valid_split(full_dset, split_fold=10, random_seed=args.dataset_seed)
    train_loader, val_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size_train, shuffle=True), torch.utils.data.DataLoader(val_dset, batch_size=args.batch_size_val, shuffle=True)

    # We create a configuration file with all the parameters
    model_name = 'repulsive_train:{}_repulsive:{}_lambda:{}_bandwidth:{}'.format(args.train.lower(), args.repulsive,
                                                                                 args.lambda_repulsive,
                                                                                 args.bandwidth_repulsive)
    if args.id is not None:
        model_name = model_name + '_{}'.format(args.id)

    savepath = Path(args.save_folder)
    try:
        if not Path.exists(savepath):
            os.makedirs(savepath)

        if not Path.exists(savepath / 'config.json'):  # Only create json if it does not exist
            with open(savepath / 'config.json', 'w') as fd:
                json.dump(vars(args), fd)
    except FileExistsError:
        print('File already exists')
        pass

    # If the experiment is name we save it in results directly.
    # experiment.log_parameters(vars(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    VAL_FREQ = 1

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # Load the reference net
    if args.repulsive is not None:
        if args.repulsive.lower() == 'fashionmnist':
            # For the repulsive loader we don't need to split into train and validation, we can use the full set
            tfms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.2859,), (0.3530,))])
            dset_repulsive = torchvision.datasets.FashionMNIST('../../datasets/FashionMNIST', train=True, download=False, transform=tfms)

            # Load the repulsive model
            raw_model = models.NNMNIST(28 * 28, 10)
            reference_net = tools.load_model(Path(args.reference_net), raw_model)
            reference_net.eval()

        else:
            raise ValueError('Bad repulsive dataset selected: {}'.format(args.repulsive.lower()))

        # Create repulsive sampler
        repulsive_loader = torch.utils.data.DataLoader(dset_repulsive, batch_size=args.batch_size_repulsive, shuffle=True)
        repulsive_sampler = sampler.repulsiveSampler(args.repulsive.upper(), dataloader=repulsive_loader,
                                                     batch_size=args.batch_size_repulsive)

    print('Finished loading the datasets.')

    # Partial functions
    if args.repulsive is not None:
        _optimize = partial(tools.optimize, bandwidth_repulsive=args.bandwidth_repulsive,
                            lambda_repulsive=args.lambda_repulsive)
    else:
        _optimize = tools.optimize

    # --------------------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------------------
    step = 0

    for epoch in tqdm(range(args.n_epochs), desc='epochs'):
        # Training phase
        net.train()
        _tqdm = tqdm(train_loader, desc='batch')
        # experiment.log_current_epoch(epoch)
        for j, batch_raw in enumerate(_tqdm):

            if args.repulsive is not None:
                br = repulsive_sampler.sample_batch()
                batch_repulsive = br.to(device)

            # optimization part # prepare the batch, we get images not vectors !
            x_raw, y = batch_raw
            if args.repulsive is not None:
                batch_repulsive = prepr(batch_repulsive)
            x_raw, y = prepr(x_raw), y.view(-1)
            batch = (x_raw.to(device), y.to(device))

            if args.repulsive is not None:
                kwargs = {'reference_net': reference_net, 'batch_repulsive': batch_repulsive}
            else:
                kwargs = {'beta':args.beta, 'prior': prior}
            info_training = _optimize(net, optimizer, batch, add_repulsive_constraint=args.repulsive is not None,
                                      **kwargs)
            if args.verbose:
                _tqdm.set_description(
                    'Epoch {}/{}, loss: {:.4f}'.format(epoch + 1, args.n_epochs, info_training['loss']))

            # # Log to Comet.ml
            # for k, v in info_training.items():
            #     experiment.log_metric(k, float(v), step=step)
            step += 1

        if not Path.exists(savepath / 'models'):
            os.makedirs(savepath / 'models')

        if (epoch > 0 and epoch % args.save_freq == 0):
            model_path = savepath / 'models' / '{}_{}epochs.pt'.format(model_name, epoch + 1)
            if not Path.exists(model_path):
                torch.save(net.state_dict(), model_path)
            else:
                raise ValueError('Error trying to save file at location {}: File already exists'.format(model_path))

        if epoch % VAL_FREQ == 0:

            # Evaluate on validation set
            xent = nn.CrossEntropyLoss()
            net.eval()
            total_val_loss, total_val_acc = 0.0, 0.0
            n_val = len(val_loader.dataset)

            for j, batch_raw in enumerate(val_loader):
                x_raw, y = batch_raw
                len_batch = x_raw.size(0)
                x_raw, y = prepr(x_raw), y.view(-1)
                x, y = x_raw.to(device), y.to(device)
                y_logit = net(x)

                # logging
                total_val_loss += (len_batch / n_val) * xent(y_logit, y.view(-1)).item()
                total_val_acc += (y_logit.argmax(1) == y).float().sum().item() / n_val

            # Compute statistics
            print('Epoch {}/{}, val acc: {:.3f}, val loss: {:.3f}'.format(epoch + 1, args.n_epochs, total_val_acc,
                                                                          total_val_loss))
            # experiment.log_metric("val_accuracy", total_val_acc)
            # experiment.log_metric("val_loss", total_val_loss)

    # POST-PROCESSING
    # Save the model
    try:
        dirname = 'models-beta:{}'.format(args.beta)
        if not Path.exists(savepath / dirname):
            os.makedirs(savepath / dirname)

        model_path = savepath / dirname / '{}_{}epochs.pt'.format(model_name, epoch + 1)
        if not Path.exists(model_path):
            torch.save(net.state_dict(), model_path)
    except FileExistsError:
        print('Error trying to save file at location {}: File already exists')

def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--train', required=True, type=str)
    parser.add_argument('--repulsive', type=str)
    parser.add_argument('--reference_net', type=str, default='log/repulsive_default/reference_net_mnist.model')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_repulsive', type=float, default=0.0)
    parser.add_argument('--bandwidth_repulsive', type=float, default=0.0)
    parser.add_argument('--batch_size_train', type=int, default=64)
    parser.add_argument('--batch_size_val', type=int, default=64)
    parser.add_argument('--batch_size_repulsive', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--id', type=int)
    parser.add_argument('--save_folder', type=str, default='log/mnist')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--comet', action='store_true')
    parser.add_argument('--beta', type=float, default=0.0)

    args = parser.parse_args()
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    # Log the experiment
    # experiment = Experiment(api_key="XXXXX", project_name="XXXXX", workspace="XXXXX",
    #                         disabled=not args.comet)
    # experiment.train()
    train(args, None, device)


if __name__ == '__main__':
    main()
