import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
import functools


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="bilinear"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)


class Generator(nn.Module):
    def __init__(self, channels, dim):
        super(Generator, self).__init__()
        self.channels = channels
        self.dim = dim

        self.x_down1 = self.down_block(self.channels, self.dim, normalization=False)  # 256 -> 128  3 -> 64
        self.x_down2 = self.down_block(self.dim, self.dim * 2)  # 128 -> 64   64 -> 128
        self.x_down3 = self.down_block(self.dim * 2, self.dim * 4)  # 64 -> 32  128 -> 256
        self.x_down4 = self.down_block(self.dim * 4, self.dim * 8)  # 32 -> 16  256 -> 512
        self.x_down5 = self.down_block(self.dim * 8, self.dim * 8)  # 16 -> 8  512 -> 512
        self.x_down6 = self.down_block(self.dim * 8, self.dim * 8)  # 8 -> 4  512 -> 512
        self.x_down7 = self.down_block(self.dim * 8, self.dim * 8)  # 4 -> 2  512 -> 512
        self.x_down8 = self.down_block(self.dim * 8, self.dim * 8, dropout=0.5,
                                       normalization=False)  # 2 -> 1  512 -> 1024

        # up
        self.x_up1 = self.up_block(self.dim * 8, self.dim * 8, dropout=0.5, normalization=False)  # 1 -> 2
        self.x_up2 = self.up_block(self.dim * 16, self.dim * 8, dropout=0.5)  # 2 -> 4
        self.x_up3 = self.up_block(self.dim * 16, self.dim * 8, dropout=0.5)  # 4 -> 8
        self.x_up4 = self.up_block(self.dim * 16, self.dim * 8)  # 8 -> 16
        self.x_up5 = self.up_block(self.dim * 16, self.dim * 4)  # 16 -> 32
        self.x_up6 = self.up_block(self.dim * 8, self.dim * 2)  # 32 -> 64
        self.x_up7 = self.up_block(self.dim * 4, self.dim)  # 64 -> 128
        self.x_up8 = self.up_block(self.dim * 2, self.dim * 2)
        self.out_conv = nn.Conv2d(self.dim * 2, self.channels, 3, 1, 1)

        self.tanh = nn.Tanh()

        self.G_loss = []

    @staticmethod
    def upSample_block(input_channels, output_channels, scale_factor=2, kernel_size=3, stride=1, padding=1,
                       normalization=True, dropout=0.):
        block = [Upsample(scale_factor),
                 nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, padding_mode="reflect")]
        if normalization:
            block.append(nn.BatchNorm2d(output_channels))
        block.append(nn.ReLU(True))
        if dropout:
            block.append(nn.Dropout(dropout))
        return nn.Sequential(*block)

    @staticmethod
    def down_block(input_channels, output_channels, kernel_size=3, stride=2, padding=1, normalization=True, dropout=0.):
        block = [
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, padding_mode="reflect",
                      bias=not normalization),
        ]
        if normalization:
            block.append(nn.BatchNorm2d(output_channels))
        block.append(nn.LeakyReLU(0.2, True))
        if dropout:
            block.append(nn.Dropout(dropout))

        return nn.Sequential(*block)

    @staticmethod
    def up_block(input_channels, output_channels, kernel_size=4, stride=2, padding=1,
                 normalization=True, dropout=0.):
        block = [
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding,
                               bias=not normalization)
        ]
        if normalization:
            block.append(nn.BatchNorm2d(output_channels))

        block.append(nn.ReLU(True))

        if dropout:
            block.append(nn.Dropout(dropout))

        return nn.Sequential(*block)

    def forward(self, x):
        # down
        x_down1 = self.x_down1(x)  # 3 -> 64
        x_down2 = self.x_down2(x_down1)
        x_down3 = self.x_down3(x_down2)
        x_down4 = self.x_down4(x_down3)
        x_down5 = self.x_down5(x_down4)
        x_down6 = self.x_down6(x_down5)
        x_down7 = self.x_down7(x_down6)
        x_down8 = self.x_down8(x_down7)

        # up
        x = self.x_up1(x_down8)
        x = torch.cat((x_down7, x), dim=1)
        x = self.x_up2(x)
        x = torch.cat((x_down6, x), dim=1)
        x = self.x_up3(x)
        x = torch.cat((x_down5, x), dim=1)
        x = self.x_up4(x)
        x = torch.cat((x_down4, x), dim=1)
        x = self.x_up5(x)
        x = torch.cat((x_down3, x), dim=1)
        x = self.x_up6(x)
        x = torch.cat((x_down2, x), dim=1)
        x = self.x_up7(x)
        x = torch.cat((x_down1, x), dim=1)
        x = self.x_up8(x)

        x = self.out_conv(x)
        x = self.tanh(x)

        return x

    def plot_loss(self):
        df = pd.DataFrame(self.G_loss, columns=['discriminator loss'])
        ax = df.plot(ylim=(0, 5), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 1, 2, 3, 4, 5))
        fig = ax.get_figure()
        fig.savefig('generator_loss.png')

    def collect_loss(self, loss):
        self.G_loss.append(loss)


class Discriminator(nn.Module):
    def __init__(self, channels, dim, normalization=True):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.dim = dim
        self.normalization = normalization
        # 使得感受野为70x70, 最终的维度30x30
        self.disc = nn.Sequential(
            *self.disc_block(self.channels * 2, self.dim, normalization=False),
            *self.disc_block(self.dim, self.dim * 2),
            *self.disc_block(self.dim * 2, self.dim * 4),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(self.dim * 4, self.dim * 8, 4, 1),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(self.dim * 8, 1, 4, 1),
        )

        self.D_loss = []

    @staticmethod
    def disc_block(input_channels, output_channels, kernel_size=4, stride=2, padding=1, normalization=True):
        layer = [nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, padding_mode='reflect')]
        if normalization:
            layer.append(nn.BatchNorm2d(output_channels))
        layer.append(nn.LeakyReLU(0.2, inplace=True))
        return layer

    def forward(self, image, mask):
        x = torch.cat((image, mask), dim=1)
        return self.disc(x)

    def plot_loss(self):
        df = pd.DataFrame(self.D_loss, columns=['discriminator loss'])
        ax = df.plot(ylim=(0, 5), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 1, 2, 3, 4, 5))
        fig = ax.get_figure()
        fig.savefig('discriminator_loss.png')

    def collect_loss(self, loss):
        self.D_loss.append(loss)
