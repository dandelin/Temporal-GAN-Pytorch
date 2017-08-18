from model import TemporalGenerator, ImageGenerator, Discriminator_E, Discriminator_D
from loader import MovingMNIST

import os

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable

def reset_grad(optims):
    for optim in optims:
        optim.zero_grad()

def to_gif(vs, name, how_many=2):
    # var = N x C x D x H x W
    vs = vs.cpu().data.numpy()
    vs = np.einsum('ijklm->iklmj', vs)
    os.makedirs('outputs', exist_ok=True)

    for i, v in enumerate(vs):
        v = (v + 1) * 127.5
        v = v.astype(np.uint8)
        imageio.mimsave(f'outputs/{epoch}_{batch_idx}_{i}_{name}.gif', v)
        if i == (how_many - 1): break


if __name__ == '__main__':
    batch_size = 16
    TG = TemporalGenerator().cuda()
    IG = ImageGenerator(1).cuda()
    DE = Discriminator_E(1).cuda()
    DD = Discriminator_D(1).cuda()

    def D(X):
        X_recon = DD(DE(X))
        return torch.mean(torch.sum(torch.abs(X - X_recon), 1))

    mm_dataset = MovingMNIST()
    start_epoch = 0
    lr = 1e-3

    optim_TG = Adam(TG.parameters(), lr=lr)
    optim_IG = Adam(IG.parameters(), lr=lr)
    optim_DE = Adam(DE.parameters(), lr=lr)
    optim_DD = Adam(DD.parameters(), lr=lr)
    optims = [optim_TG, optim_IG, optim_DE, optim_DD]

    if os.path.isfile('checkpoint.pth'):
        print("=> loading checkpoint")
        checkpoint = torch.load('checkpoint.pth')
        start_epoch = checkpoint['epoch']
        TG.load_state_dict(checkpoint['TG'])
        IG.load_state_dict(checkpoint['IG'])
        DE.load_state_dict(checkpoint['DE'])
        DD.load_state_dict(checkpoint['DD'])
        optim_TG.load_state_dict(checkpoint['optim_TG'])
        optim_IG.load_state_dict(checkpoint['optim_IG'])
        optim_DE.load_state_dict(checkpoint['optim_DE'])
        optim_DD.load_state_dict(checkpoint['optim_DD'])

    for epoch in range(start_epoch, 1000000):

        mm_loader = DataLoader(mm_dataset, batch_size=batch_size, shuffle=True)

        k = 0
        lam = 1e-3
        gamma = 0.5

        for batch_idx, data in enumerate(mm_loader):
            # Sample Data
            X = Variable(data.cuda())
            
            # Discriminator
            z0_D = Variable(torch.rand(batch_size, 100, 1).cuda())
            z1_D = TG(z0_D)
            Fake_D = IG(z0_D, z1_D)
            D_X = D(X)
            D_Fake_D = D(Fake_D)

            L_D = D_X - k * D_Fake_D

            L_D.backward()
            optim_DE.step()
            optim_DD.step()
            reset_grad(optims)

            # Generator
            z0_G = Variable(torch.rand(batch_size, 100, 1).cuda())
            z1_G = TG(z0_G)
            Fake_G = IG(z0_G, z1_G)
            D_Fake_G = D(Fake_G)

            L_G = D_Fake_G

            L_G.backward()
            optim_TG.step()
            optim_IG.step()
            reset_grad(optims)

            # Update k, the equlibrium
            k = k + lam * (gamma * D_X - D_Fake_G)
            k = k.data[0] # Dismiss Variable

            measure = D_X + torch.abs(gamma * D_X - D_Fake_G)
            print(f'Epoch-{epoch}, Batch-{batch_idx}, Convergence measure: {measure.data[0]:.4}')


            if batch_idx % 100 == 0:
                to_gif(Fake_G, 'fake_g')
                to_gif(DD(DE(Fake_G)), 'fake_g_autoencoded')
                to_gif(X, 'real')
                to_gif(DD(DE(X)), 'real_autoencoded')

        torch.save({
            'epoch': epoch,
            'TG': TG.state_dict(),
            'IG': IG.state_dict(),
            'DE': DE.state_dict(),
            'DD': DD.state_dict(),
            'optim_TG': optim_TG.state_dict(),
            'optim_IG': optim_IG.state_dict(),
            'optim_DE': optim_DE.state_dict(),
            'optim_DD': optim_DD.state_dict(),
        }, 'checkpoint.pth')