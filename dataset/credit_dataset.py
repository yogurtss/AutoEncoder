import pandas as pd
import os
import librosa as lib
from tqdm import tqdm
import numpy as np
import torch
from sortedcontainers import SortedList
import h5py
import torch.nn as nn


class CreditDataset(nn.Module):
    def __init__(self, data):
        super(CreditDataset, self).__init__()
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]
