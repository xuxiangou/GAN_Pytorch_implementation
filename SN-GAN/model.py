import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
import pandas as pd
from Regularization import SpectralNorm


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class Generator(nn.Module):
    def __init__(self, latent_dim, batch_size, dim, channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.dim = dim
        self.channels = channels
        self.net = nn.Sequential(
            # View(self.latent_dim * self.batch_size),
            nn.Linear(self.latent_dim, 4 * 4 * 4 * self.dim),
            View((-1, dim * 4, 4, 4)),
            *self.block(self.dim * 4, self.dim * 2, 4, 2, 1),  # 4 -> 8
            *self.block(self.dim * 2, self.dim, 4, 2, 1),  # 8 -> 16
            *self.block(self.dim, self.dim, 4, 2, 1),  # 16 -> 32
            nn.Conv2d(self.dim, channels, 3, 1, 1),  # 16 -> 32
            nn.Tanh(),
        )

        self.G_loss = []

    @staticmethod
    def block(in_dim, out_dim, kernel_size, stride, padding=0):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, stride=stride,
                               padding=padding),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        return block

    def forward(self, x):
        x = self.net(x)
        return x

    def collect_loss(self, loss):
        self.G_loss.append(loss)

    def plot_loss(self):
        df = pd.DataFrame(self.G_loss, columns=['discriminator loss'])
        ax = df.plot(ylim=(0, 100.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 1, 2))
        fig = ax.get_figure()
        fig.savefig('generator_loss.png')


# Discriminator中不使用BatchNormalization
class Discriminator(nn.Module):
    def __init__(self, latent_dim, batch_size, dim, channels):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.dim = dim
        self.channels = channels

        self.net = nn.Sequential(
            *self.block(self.channels, self.dim, 3, 1, 1),
            *self.block(self.dim, self.dim, 4, 2, 1),  # 32 -> 16
            *self.block(self.dim, self.dim * 2, 3, 1, 1),
            *self.block(self.dim * 2, self.dim * 2, 4, 2, 1),  # 16 -> 8
            *self.block(self.dim * 2, self.dim * 4, 3, 1, 1),
            *self.block(self.dim * 4, self.dim * 4, 4, 2, 1),  # 8 -> 4
            View((-1, self.dim * 4 * 4 * 4)),
            SpectralNorm(nn.Linear(self.dim * 4 * 4 * 4, 1)),  # 4 -> 1
        )

        self.D_loss = []

    @staticmethod
    def block(in_dim, out_dim, kernel_size, stride, padding=0):
        block = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, stride=stride,
                                    padding=padding)),
            nn.LeakyReLU(0.1, inplace=True),
        )
        return block

    def forward(self, x):
        x = self.net(x)
        return x

    def collect_loss(self, loss):
        self.D_loss.append(loss)

    def plot_loss(self):
        df = pd.DataFrame(self.D_loss, columns=['discriminator loss'])
        ax = df.plot(ylim=(0, 100.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 1, 2))
        fig = ax.get_figure()
        fig.savefig('discriminator_loss.png')