'''
This files creates a csv comparing all the models in a given folder.
'''
import h5py
import pandas as pd
from tqdm import tqdm
import argparse as ap
import numpy as np
from pathlib import Path
import os

# Creating the parser to use parameters
parser = ap.ArgumentParser()
parser.add_argument('--folder', type=str, required=True, help='Folders to consider when creating the figures')
parser.add_argument('--prefix', type=str, help='prefix to use for folders', required=True)
parser.add_argument('--test', type=str, default='mnist')
parser.add_argument('--entropy_evaluation', type=str, default='notmnist')
parser.add_argument('--n_nets', default=[10], nargs='+', required=False)
parser.add_argument('--save', action='store_true')

args = parser.parse_args()

filenames = ['baseline']
results = np.array([[0, 0, 0, 0]])


def work(p, dirname, args):
    r'''
    Here we want to compute the entropies and KL divergences for all experiments, wrt the baseline
    '''

    with h5py.File(p / 'stats/{}.h5'.format(args.test), 'r') as fd:
        entro_test_experiment_vals = fd.get('std')[:]
    with h5py.File(p / 'stats/{}.h5'.format(args.entropy_evaluation), 'r') as fd:
        entro_evaluation_experiment_vals = fd.get('std')[:]

    # We compute the mean entropy on the test and evaluation datasets for our experiment
    mean_entro_experiment_test = np.mean(entro_test_experiment_vals)
    mean_entro_experiment_evaluation = np.mean(entro_evaluation_experiment_vals)

    # Now we compute the standard deviation of the entropy on the test and evaluation datsets for our experiment
    std_entro_experiment_test = np.std(entro_test_experiment_vals)
    std_entro_experiment_evaluation = np.std(entro_evaluation_experiment_vals)

    # Return the values
    return np.array([[mean_entro_experiment_test, std_entro_experiment_test, mean_entro_experiment_evaluation, std_entro_experiment_evaluation]])


# ===============================================================================
# Going through the files and creating the figures
# ===============================================================================

# loop over all folders and if there is no std.h5 file we create it
folder = Path(args.folder)

for dirname in tqdm(os.listdir(folder)):
    if os.path.isdir(folder / dirname):
        if dirname[:len(args.prefix)] == args.prefix:
            result = work(folder / dirname, dirname, args)
            results = np.concatenate((results, result), axis=0)
            filenames.append(dirname)

# Finally we save a csv file
df = pd.DataFrame(results, index=filenames)
df.to_csv(folder / 'stats_{}.csv'.format(args.prefix))
