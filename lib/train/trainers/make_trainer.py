from .trainer import Trainer
import imp
import os
from lib.config import cfg


def _wrapper_factory(cfg, network):
    module = '.'.join(['lib.train.trainers', cfg.task])
    if cfg.isrec and cfg.ifmultistage:
        path = os.path.join('lib/train/trainers', cfg.task+'rec.py')
    else:
        path = os.path.join('lib/train/trainers', cfg.task+'.py')
    network_wrapper = imp.load_source(module, path).NetworkWrapper(network)
    return network_wrapper


def make_trainer(cfg, network):
    network = _wrapper_factory(cfg, network)
    return Trainer(network)
