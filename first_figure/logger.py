from pathlib import Path
import os
import json
import pickle
import sys
from datetime import datetime
import torch
import random


class JsonLogger():
    def __init__(self, log_dir='./log', verbose=0):
        self.verbose = verbose
        self.log_dir = Path(log_dir)
        self.current_logdir = None
        self.log_name = None
        self.closed_log = False

        if verbose > 0:
            print('ModelLogger created!')

    def _check_log_directory(self, directory):
        """
        Check that the log directory exists and create it if it doesn't.
        Args:
            directory: log directory path.
        Return:
        """

        try:
            if not Path.exists(directory):
                self.print("Attempting to make log directory at {0}".format(directory))
                os.makedirs(directory)
        except IOError as e:
            sys.exit("Error attempting to create log directory: {0}".format(e.strerror).strip())

    def new_log(self, logname=None):
        """
        :param initial_dict:
        :return:
        """
        assert not self.closed_log, 'log already closed'
        random.seed()
        rnd_id = random.randint(0, 100000)
        if logname is not None:
            self.log_name = 'tmp_' + logname + '_' + str(rnd_id)
        else:
            self.log_name = 'tmp_' + (datetime.now().strftime('%d-%m-%Y_%H-%M-%S')) + '_' + str(rnd_id)
        self.current_logdir = self.log_dir / self.log_name

        self._check_log_directory(self.log_dir / self.log_name)

    def add_json(self, filename, dict_to_add_as_json):
        assert not self.closed_log, 'log already closed'
        with open(self.current_logdir / filename, 'w') as f:
            json.dump(dict_to_add_as_json, f)

    def add_pickle(self, filename, object_to_add):
        assert not self.closed_log, 'log already closed'
        create_dir_if_needed(self, 'pkl')
        with open(self.current_logdir / 'pkl' / filename, 'wb') as f:
            pickle.dump(object_to_add, f)

    def add_model(self, filename, torch_model):
        assert not self.closed_log, 'log already closed'
        create_dir_if_needed(self, 'models')
        torch.save(torch_model.state_dict(), self.current_logdir / 'models' / filename)

    def print(self, message):
        assert type(message) == str
        timeinfo = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        print(timeinfo + ' -- ' + message)

    def close_log(self):
        assert not self.closed_log, 'log already closed'
        final_name_dir = str(self.log_name[4:])
        # We rename the directory
        os.rename(self.log_dir / self.log_name, self.log_dir / final_name_dir)
        self.print('Renaming log from {} to {}'.format(self.log_name, final_name_dir))
        # os.rename(self.log_name, final_name_dir)
        self.closed_log = True


def create_dir_if_needed(logger, dir_name):
    """
    Creates a directory to save files (images or pickle files for instance) if it does not already exist.
    :param logger:
    :param dir_name:
    :return:
    """
    directory_save = logger.current_logdir / dir_name
    if not os.path.exists(directory_save):
        logger.print("Attempting to make log directory at {0}".format(directory_save))
        os.makedirs(directory_save)


def open_json(path):
    try:
        with open(path, 'r') as fd:
            content = json.load(fd)
    except FileNotFoundError:
        print('Did not find the requested json file at {0}.'.format(path))
        return -1
    return content
