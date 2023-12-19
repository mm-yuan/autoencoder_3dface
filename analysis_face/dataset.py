import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
import h5py
import scipy.io as sio
from scipy.io import loadmat


class FaceDataset(Dataset):

    def __init__(self, args):
        tmp = h5py.File(args.data_dir)

        self.shapes = np.float32(np.transpose(tmp["faces"], (0, 2, 1)))
        if args.normalization:
            self.mean = np.mean(self.shapes, axis=0)
            # self.std = np.std(self.shapes, axis=0)
            self.shapes -= self.mean
            # self.shapes/= self.std
        tmp = sio.loadmat(args.sets_ind)

    def __len__(self):
        return len(self.shapes)

    def __getitem__(self, idx):
        shape = np.squeeze(self.shapes[idx, :, :])
        return shape



