import os
import os.path as osp
import numpy as np
from matplotlib import pyplot as plt
import torch

from trainer.base import BaseTrainer
from datasets import get_dataset
from models import get_model

class Trainer(BaseTrainer):
    def prepare_dataloader(self, data_config):
        pass

    def build_model(self, model_config):
        pass

    def set_loss_function(self, loss_config):
        pass

    def forward(self, data):
        pass

    def train_step(self, data):
        pass

    def val_step(self, data):
        pass

    def visualize_batch(self):
        pass
