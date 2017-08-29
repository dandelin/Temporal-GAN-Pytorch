import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image, ImageSequence

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

class GIF(Dataset):
    def __init__(self, path='gifs'):
        self.paths = glob(f'{path}/*.gif')
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        path = self.paths[index]
        tensors = torch.zeros(16, 3, 64, 64)
        try:
            gif = Image.open(path)
            gif_iter = ImageSequence.Iterator(gif)
            gifs = [img.convert('RGB') for img in gif_iter]

            f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
            indice = f(16, len(gifs))
            frames = [gifs[i] for i in indice]

            for i, frame in enumerate(frames):
                h, w = frame.size
                hpad = (max(h, w) - h) // 2
                wpad = (max(h, w) - w) // 2
                pad = (hpad, wpad, hpad, wpad)
                transform = transforms.Compose([
                    transforms.Pad(pad),
                    transforms.Scale([64, 64]),
                    transforms.ToTensor()
                ])
                image_tensor = transform(frame)
                tensors[i, :, :, :] = image_tensor
            tensors = torch.transpose(tensors, 0, 1)
            tensors = tensors * 2 - 1
        except:
            pass
        return tensors

if __name__ == '__main__':
    dset = GIF()
    loader = DataLoader(dset, batch_size=2)
    for data in loader:
        pass
