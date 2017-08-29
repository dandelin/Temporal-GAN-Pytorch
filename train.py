from model import TemporalGenerator, ImageGenerator, Discriminator_E
from loader import GIF

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
    G_TG = TemporalGenerator().cuda()
    G_IG = ImageGenerator(3).cuda()
    D_E = Discriminator_E(3).cuda()
    D_TG = TemporalGenerator().cuda()
    D_IG = ImageGenerator(3).cuda()

    def D(X):
        DE = D_E(X)
        DTG = D_TG(DE)
        DIG = D_IG(DE, DTG)
        X_recon = DIG
        return torch.mean(torch.sum(torch.abs(X - X_recon), 1))

    mm_dataset = GIF()
    start_epoch = 0
    lr = 1e-3

    optim_G_TG = Adam(G_TG.parameters(), lr=lr)
    optim_G_IG = Adam(G_IG.parameters(), lr=lr)
    optim_D_E = Adam(D_E.parameters(), lr=lr)
    optim_D_TG = Adam(D_TG.parameters(), lr=lr)
    optim_D_IG = Adam(D_IG.parameters(), lr=lr)
    optims = [optim_G_TG, optim_G_IG, optim_D_E, optim_D_TG, optim_D_IG]

    if os.path.isfile('checkpoint.pth'):
        print("=> loading checkpoint")
        checkpoint = torch.load('checkpoint.pth')
        start_epoch = checkpoint['epoch']
        G_TG.load_state_dict(checkpoint['G_TG'])
        G_IG.load_state_dict(checkpoint['G_IG'])
        D_E.load_state_dict(checkpoint['D_E'])
        D_TG.load_state_dict(checkpoint['D_TG'])
        D_IG.load_state_dict(checkpoint['D_IG'])
        optim_G_TG.load_state_dict(checkpoint['optim_G_TG'])
        optim_G_IG.load_state_dict(checkpoint['optim_G_IG'])
        optim_D_E.load_state_dict(checkpoint['optim_D_E'])
        optim_D_TG.load_state_dict(checkpoint['optim_D_TG'])
        optim_D_IG.load_state_dict(checkpoint['optim_D_IG'])

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
            z1_D = G_TG(z0_D)
            Fake_D = G_IG(z0_D, z1_D)
            D_X = D(X)
            D_Fake_D = D(Fake_D)

            L_D = D_X - k * D_Fake_D

            L_D.backward()
            optim_D_E.step()
            optim_D_TG.step()
            optim_D_IG.step()
            reset_grad(optims)

            # Generator
            z0_G = Variable(torch.rand(batch_size, 100, 1).cuda())
            z1_G = G_TG(z0_G)
            Fake_G = G_IG(z0_G, z1_G)
            D_Fake_G = D(Fake_G)

            L_G = D_Fake_G

            L_G.backward()
            optim_G_TG.step()
            optim_G_IG.step()
            reset_grad(optims)

            # Update k, the equlibrium
            k = k + lam * (gamma * D_X - D_Fake_G)
            k = k.data[0] # Dismiss Variable

            measure = D_X + torch.abs(gamma * D_X - D_Fake_G)
            print(f'Epoch-{epoch}, Batch-{batch_idx}, Convergence measure: {measure.data[0]:.4}')


            if batch_idx % 100 == 0:
                to_gif(Fake_G, 'fake_g')
                to_gif(D_IG(D_E(Fake_G), D_TG(D_E(Fake_G))), 'fake_g_autoencoded')
                to_gif(X, 'real')
                to_gif(D_IG(D_E(X), D_TG(D_E(X))), 'real_autoencoded')

        torch.save({
            'epoch': epoch,
            'G_TG': G_TG.state_dict(),
            'G_IG': G_IG.state_dict(),
            'D_E': D_E.state_dict(),
            'D_TG': D_TG.state_dict(),
            'D_IG': D_IG.state_dict(),
            'optim_G_TG': optim_G_TG.state_dict(),
            'optim_G_IG': optim_G_IG.state_dict(),
            'optim_D_E': optim_D_E.state_dict(),
            'optim_D_TG': optim_D_TG.state_dict(),
            'optim_D_IG': optim_D_IG.state_dict(),
        }, 'checkpoint.pth')
