import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class TemporalGenerator(nn.Module):
    """
        (#batch, #channel=100, #length=1) : z0
        --> (#batch, #channel=100, #length=16)
    """
    def __init__(self):
        super(TemporalGenerator, self).__init__()
        # out = (in − 1) ∗ stride − 2 ∗ padding + kernel_size + output_padding
        self.deconv1 = nn.ConvTranspose1d(100, 512, 1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(512)
        self.deconv2 = nn.ConvTranspose1d(512, 256, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.deconv3 = nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.deconv4 = nn.ConvTranspose1d(128, 128, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.deconv5 = nn.ConvTranspose1d(128, 100, 4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm1d(100)
    def forward(self, z0):
        ret = F.relu(self.bn1(self.deconv1(z0)))
        ret = F.relu(self.bn2(self.deconv2(ret)))
        ret = F.relu(self.bn3(self.deconv3(ret)))
        ret = F.relu(self.bn4(self.deconv4(ret)))
        ret = F.tanh(self.bn5(self.deconv5(ret)))
        return ret

class ImageGenerator(nn.Module):
    """
        (#batch, #channel=100, #length=1) : z0
        (#batch, #channel=100, #length=16) : z1
        --> (#batch, #channel=3, #h=64, #w=64)
    """
    def __init__(self, channel):
        super(ImageGenerator, self).__init__()
        self.linear_z0 = nn.Linear(100, 256 * 4 * 4)
        self.linear_z1 = nn.Linear(100, 256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.deconv5 = nn.ConvTranspose2d(32, channel, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(channel)
        
    def forward(self, z0, z1):
        z0_b, z0_c, z0_l = z0.size()
        z1_b, z1_c, z1_l = z1.size()
        z0 = self.linear_z0(z0.transpose(2, 1).contiguous().view(z0_b * z0_l, 100)).contiguous().view(z0_b, z0_l, 256, 4, 4)
        z1 = self.linear_z1(z1.transpose(2, 1).contiguous().view(z1_b * z1_l, 100)).contiguous().view(z1_b, z1_l, 256, 4, 4)
        ret = []
        for t in range(z1_l):
            z = torch.cat([z0[:, 0, :, :, :], z1[:, t, :, :, :]], dim=1)
            z = F.relu(self.bn1(self.deconv1(z)))
            z = F.relu(self.bn2(self.deconv2(z)))
            z = F.relu(self.bn3(self.deconv3(z)))
            z = F.relu(self.bn4(self.deconv4(z)))
            z = F.tanh(self.bn5(self.deconv5(z)))
            ret.append(z)
        ret = torch.stack(ret) # D x N x C x H x W
        ret = ret.transpose(0, 1).transpose(1, 2) # N x C x D x H x W
        return ret

class Discriminator_E(nn.Module):
    def __init__(self, channel):
        super(Discriminator_E, self).__init__()
        self.conv1 = nn.Conv3d(channel, 64, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 512, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(512)
        self.fc = nn.Linear(512 * 4 * 4, 100)

    def forward(self, gz):
        gz = F.leaky_relu(self.bn1(self.conv1(gz)), negative_slope=0.2)
        gz = F.leaky_relu(self.bn2(self.conv2(gz)), negative_slope=0.2)
        gz = F.leaky_relu(self.bn3(self.conv3(gz)), negative_slope=0.2)
        gz = F.leaky_relu(self.bn4(self.conv4(gz)), negative_slope=0.2)
        gz = F.tanh(self.fc(gz.view(-1, 512 * 4 * 4)))
        gz = gz.view(-1, 100, 1)
        return gz

if __name__ == '__main__':
    tg = TemporalGenerator()
    ig = ImageGenerator()
    de = Discriminator_E()
    dd = Discriminator_D()
    z0 = Variable(torch.rand(1, 100, 1))
    z1 = tg(z0)
    frames = ig(z0, z1)
    h = de(frames)
    eh = dd(h)
    print(eh)