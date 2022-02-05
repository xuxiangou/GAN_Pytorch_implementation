import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Residual_block_up(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding):
        super(Residual_block_up, self).__init__()
        self.upsample = Upsample(scale_factor=2)
        self.conv = nn.Conv2d(channels, channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = self.upsample(x)
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.conv(out)
        out = self.batch_norm(out)

        out += x

        return out


class Generator(nn.Module):
    def __init__(self, latent_dim, batch_size, dim, channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.dim = dim
        self.channels = channels
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, 4 * 4 * self.dim),
            View((self.batch_size, dim, 4, 4)),
            Residual_block_up(self.dim, 3, 1, 1),  # 4 -> 8
            Residual_block_up(self.dim, 3, 1, 1),  # 8 -> 16
            Residual_block_up(self.dim, 3, 1, 1),  # 16 -> 32
            nn.Conv2d(self.dim, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

        self.G_loss = []

    def forward(self, x):
        x = self.net(x)
        return x

    def collect_loss(self, loss):
        self.G_loss.append(loss)

    def plot_loss(self):
        plt.plot(self.G_loss)
        plt.show()
        plt.savefig('G_loss.png')


class Residual_block_down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, connected=True, down=True):
        super(Residual_block_down, self).__init__()
        self.mean_pool = nn.AvgPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.connected = connected
        self.down = down

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.connected:
            out += x

        if self.down:
            out = self.mean_pool(out)

        return out


# Discriminator中不使用BatchNormalization
class Discriminator(nn.Module):
    def __init__(self, latent_dim, batch_size, dim, channels):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.dim = dim
        self.channels = channels

        self.net = nn.Sequential(
            Residual_block_down(3, self.dim, 3, 1, 1, connected=False),  # 32 -> 16
            Residual_block_down(self.dim, self.dim, 3, 1, 1),  # 16 -> 8
            Residual_block_down(self.dim, self.dim, 3, 1, 1),  # 8 -> 4
            Residual_block_down(self.dim, self.dim, 3, 1, 1, down=False),  # 4 -> 4
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4),
            View((self.batch_size, -1)),
            nn.Linear(self.dim, 1),
        )

        self.D_loss = []

    def forward(self, x):
        x = self.net(x)
        return x

    def collect_loss(self, loss):
        self.D_loss.append(loss)

    def plot_loss(self):
        plt.plot(self.D_loss)
        plt.show()
        plt.savefig('D_loss.png')
