import numpy as np
from torch.utils.data import Dataset

class MovingMNIST(Dataset):
    def __init__(self, path='mnist_test_seq.npy'):
        self.data = np.load(path) # 20 x 10000 x 64 x 64
    def __len__(self):
        return self.data.shape[1]
    def __getitem__(self, index):
        ret = self.data[:16, index, :, :] # D x H x W
        ret = ret.astype(np.float32)
        ret = (ret / 255) * 2 - 1
        # C x D x H x W
        return np.expand_dims(ret, axis=0)