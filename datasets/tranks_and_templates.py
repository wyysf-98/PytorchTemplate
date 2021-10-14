import os
import os.path as osp
import json
import cv2
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class Dataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = dict()

        sample['idx'] = idx
          
        return sample