"""
for this model, it is the PG-GAN which used to generate high revolution image

Due to the restrict of our GPU, we just generate image of 256x256
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_network import PGLinear, PGConv, pixelnorm


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


def miniBatchStddev(x, subGroupSize=4):
    size = x.shape
    subGroupSize = min(size[0], subGroupSize)
    if size[0] % subGroupSize != 0:
        subGroupSize = size[0]

    G = int(size[0] / subGroupSize)

    if subGroupSize > 1:
        y = x.view(-1, subGroupSize, size[1], size[2], size[3])
        # 计算方差
        y = torch.var(y, 1)
        # eps = 1e-8
        y = torch.sqrt(y + 1e-8)
        y = y.view(G, -1)
        y = torch.mean(y, 1).view(G, 1)
        y = y.expand(G, size[2] * size[3]).view((G, 1, 1, size[2], size[3]))
        y = y.expand(G, subGroupSize, -1, -1, -1)
        y = y.contiguous().view((-1, 1, size[2], size[3]))

    else:
        y = torch.zeros((size[0], 1, size[2], size[3]))

    return torch.cat((x, y), dim=1)


class discriminator_block(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(discriminator_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.cov3x3_1 = PGConv(self.in_channels, self.out_channels, 3, 1, device=self.device)
        self.cov3x3_2 = PGConv(self.out_channels, self.out_channels, 3, 1, device=self.device)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.cov3x3_1(x)
        x = self.leaky_relu(x)
        x = self.cov3x3_2(x)
        x = self.leaky_relu(x)
        x = self.avg_pool(x)

        return x


# no pixel normalization
class Discriminator(nn.Module):
    def __init__(self, latent_dim, channels, device, normalization=True):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.device = device

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.from_RGB = nn.ModuleList()
        self.scale_layer = nn.ModuleList()
        self.avg_pool = nn.AvgPool2d(kernel_size=2)

        self.fin_net = nn.Sequential(
            PGConv(self.latent_dim + 1, self.latent_dim, 3, 1, device=self.device),
            nn.LeakyReLU(0.2, inplace=True),
            View((-1, self.latent_dim * 4 * 4)),
            PGLinear(self.latent_dim * 4 * 4, 1, device=self.device)
        )

        self.from_RGB.append(PGConv(self.channels, self.latent_dim, 1, 0, device=self.device))

        self.alpha = 0

    def add_layer(self, in_channels, out_channels):
        self.scale_layer.append(discriminator_block(in_channels, out_channels, device=self.device))
        self.from_RGB.append(PGConv(self.channels, in_channels, 1, 0, device=self.device))

    def update_alpha(self, alpha):
        if alpha < 0 or alpha > 1:
            raise ValueError("the alpha should be [0, 1]")

        self.alpha = alpha

    def forward(self, x):
        if self.alpha > 0 and len(self.from_RGB) > 1:
            y = self.avg_pool(x)
            y = self.leaky_relu(self.from_RGB[-2](y))

        x = self.from_RGB[-1](x)
        merge_layer = self.alpha > 0 and len(self.scale_layer) > 1

        for layer in reversed(self.scale_layer):
            x = layer(x)

            # at the beginning layer
            if merge_layer:
                merge_layer = False
                x = self.alpha * y + (1 - self.alpha) * x

        x = miniBatchStddev(x)

        x = self.fin_net(x)

        return x


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class generator_block(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(generator_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.up_sample = Upsample(scale_factor=2)
        self.cov3x3_1 = PGConv(self.in_channels, self.out_channels, 3, 1, device=self.device)
        self.cov3x3_2 = PGConv(self.out_channels, self.out_channels, 3, 1, device=self.device)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.pixelnorm = pixelnorm()

    def forward(self, x):
        x = self.up_sample(x)
        x = self.cov3x3_1(x)
        x = self.pixelnorm(x)
        x = self.leaky_relu(x)
        x = self.cov3x3_2(x)
        x = self.pixelnorm(x)
        x = self.leaky_relu(x)

        return x


class Generator(nn.Module):
    def __init__(self, latent_dim, channels, image_size, device, channels_scale=32, ):
        """
        the generator of gen
        :param latent_dim: the input channels of tensor
        :param channels: the output channels of image
        :param image_size: the size of the final image
        :param channels_scale: the scale factor of channels
        """
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.channels = channels
        self.device = device
        self.initial_net = nn.Sequential(
            PGLinear(self.latent_dim, self.latent_dim * 4 * 4, device=self.device),
            View((-1, self.latent_dim, 4, 4)),
            nn.LeakyReLU(0.2, inplace=True),
            PGConv(self.latent_dim, self.latent_dim, 3, 1, device=self.device),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.image_size = image_size
        self.up_times = int(np.log2(self.image_size / 4))
        self.channels_down_times = int(np.log2(channels_scale))
        self.upsample = Upsample(scale_factor=2)
        self.alpha = 0.

        self.scale_layer = nn.ModuleList()
        self.toRGB = nn.ModuleList()

        self.toRGB.append(PGConv(self.latent_dim, self.channels, 1, 0, device=self.device))

    def add_layer(self, ):
        if self.up_times == 0:
            raise ValueError("you cannot continue to up-sample")
        # check!
        if self.up_times - self.channels_down_times <= 0:
            in_channels = int(2 ** (self.up_times - self.channels_down_times) * self.latent_dim)
            out_channels = in_channels // 2
        else:
            in_channels = self.latent_dim
            out_channels = in_channels

        self.scale_layer.append(generator_block(in_channels, out_channels, device=self.device))
        self.toRGB.append(PGConv(out_channels, self.channels, 1, 0, device=self.device))

        self.up_times -= 1

        return in_channels, out_channels

    def out_size(self):
        return int(4 * (2 ** (len(self.toRGB) - 1)))

    def update_alpha(self, alpha):
        if alpha > 1 or alpha < 0:
            raise ValueError("alpha is [0, 1]")

        self.alpha = alpha

    def forward(self, x):
        x = self.initial_net(x)

        if self.alpha > 0 and len(self.scale_layer) == 1:
            y = self.toRGB[-2](x)
            y = self.upsample(y)

        for scale, layer in enumerate(self.scale_layer, 0):
            x = layer(x)

            # before the last upsample
            if self.alpha > 0 and scale == len(self.scale_layer) - 2:
                y = self.toRGB[-2](x)
                y = self.upsample(y)

        x = self.toRGB[-1](x)

        if self.alpha > 0 and len(self.scale_layer) >= 1:
            x = self.alpha * y + (self.alpha - 1) * x

        # no activation
        if self.image_size < 1024:
            x = nn.Tanh()(x)

        return x

# if __name__ == '__main__':
#     b = torch.randn((4, 512)).cuda()
#     G = Generator(512, 3, 256, device=torch.device('cuda')).cuda()
#     in_, out = G.add_layer()
#     a = torch.randn((4, 3, 256, 256)).cuda()
#     D = Discriminator(512, 3, device=torch.device('cuda')).cuda()
#     D.add_layer(out, in_)
#     print(D(a).shape)
#     print(G(b).shape)
