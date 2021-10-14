import yaml
import random
import numpy as np
import pandas as pd
import logging
from itertools import repeat
import collections
try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections

class TrainClock(object):
    """ Clock object to track epoch and step during training
    """
    def __init__(self):
        self.epoch = 1
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def make_checkpoint(self):
        return {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
            'step': self.step
        }

    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.step = clock_dict['step']

class WorklogLogger:
    def __init__(self, log_file):
        logging.basicConfig(filename=log_file,
                            level=logging.INFO,
                            format='%(asctime)s - %(threadName)s -  %(levelname)s - %(message)s')

        self.logger = logging.getLogger()

    def put_line(self, line):
        self.logger.info(line)

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def set_seed(random_seed):
    ''' set random seed for random, numpy and torch. '''
    print('use random seed {}.'.format(random_seed))
    # fix random seeds for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    import torch
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)


def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    Args:
        dct: dict onto which the merge is executed
        merge_dct: dct merged into dct
    Returns: dct
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and
                isinstance(merge_dct[k], collectionsAbc.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct