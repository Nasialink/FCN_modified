import numpy as np
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, set_x, set_y):
        self.set_x = set_x
        self.set_y = set_y

    def __len__(self):
        return len(self.set_y)

    def __getitem__(self, index):

        x = torch.from_numpy(self.set_x[index, :, :, :, :])
        y = torch.from_numpy(self.set_y[index, :, :, :])
        
        return x, y
    