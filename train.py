from model import TemporalGenerator, ImageGenerator, Discriminator_E, Discriminator_D
from loader import MovingMNIST

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable


def reset_grad(optims):
    for optim in optims:
        optim.zero_grad()

if __name__ == '__main__':
    batch_size = 2
    TG = TemporalGenerator().cuda()
    IG = ImageGenerator().cuda()
    DE = Discriminator_E().cuda()
    DD = Discriminator_D().cuda()

    def D(X):
        X_recon = DD(DE(X))
        return torch.mean(torch.sum(torch.abs(X - X_recon), 1))

    mm_dataset = MovingMNIST()

    for epoch in range(1000000):

        mm_loader = DataLoader(mm_dataset, batch_size=batch_size, shuffle=True)
        optim_TG = Adam(TG.parameters())
        optim_IG = Adam(IG.parameters())
        optim_DE = Adam(DE.parameters())
        optim_DD = Adam(DD.parameters())
        optims = [optim_TG, optim_IG, optim_DE, optim_DD]

        k = 0
        lam = 1e-3
        gamma = 0.5

        for batch_idx, data in enumerate(mm_loader):
            # Sample Data
            X = Variable(data.float().cuda())
            
            # Discriminator
            z0_D = Variable(torch.rand(batch_size, 100, 1).cuda())
            z1_D = TG(z0_D)
            Fake_D = IG(z0_D, z1_D)

            L_D = D(X) - k * D(Fake_D)

            L_D.backward()
            optim_DE.step()
            optim_DD.step()
            reset_grad(optims)

            # Generator
            z0_G = Variable(torch.rand(batch_size, 100, 1).cuda())
            z1_G = TG(z0_G)
            Fake_G = IG(z0_G, z1_G)

            L_G = D(Fake_G)

            L_G.backward()
            optim_TG.step()
            optim_IG.step()
            reset_grad(optims)

            # Update k, the equlibrium
            k = k + lam * (gamma * D(X) - D(Fake_G))
            k = k.data[0] # Dismiss Variable

            measure = D(X) + torch.abs(gamma * D(X) - D(Fake_G))
            print(f'Epoch-{epoch}, Batch-{batch_idx}, Convergence measure: {measure.data[0]:.4}')