import math
import torch.nn as nn
from numpy import prod


def getLayerNormalization(x):
    """
    Get He's constant for the given layer
    :param x: nn.Module
    :return: the norm value
    """
    # size: [batch_size, channels, kernel_size, kernel_size]
    size = x.weight.size()
    # channels x kernel_size x kernel_size
    fan_in = prod(size[1:])

    return math.sqrt(2 / fan_in)


class PGConv(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 device,
                 bias_init=True,
                 lrMul=1,):
        super(PGConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias_init = bias_init
        self.kernel_size = kernel_size
        self.padding = padding
        self.lrMul = lrMul
        self.device = device

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, padding=padding, device=self.device)
        self._init_weight()

    def _init_weight(self):
        # kaiming initialization
        self.conv.weight.data.normal_(0, 1)
        self.conv.weight.data /= self.lrMul
        self.weight = getLayerNormalization(self.conv)

        # initialize the bias to zero
        self.conv.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        # explicitly scale, to adjust the learning-rate
        x *= self.weight
        return x


class PGLinear(nn.Module):
    def __init__(self, in_channels, out_channels, device, bias_init=True, lrMul=1):
        super(PGLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias_init = bias_init
        self.lrMul = lrMul
        self.linear = nn.Linear(self.in_channels, self.out_channels, device=device)
        self._init_weight()

    def _init_weight(self):
        # kaiming initialization
        self.linear.weight.data.normal_(0, 1)
        self.linear.weight.data /= self.lrMul
        self.weight = getLayerNormalization(self.linear)

        # initialize the bias to zero
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = self.linear(x)
        # explicitly scale, to adjust the learning-rate
        x *= self.weight
        return x


class pixelnorm(nn.Module):
    def __init__(self, eps=1e-8):
        super(pixelnorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        # keep dim means the dim exist
        return x * (((x ** 2).mean(dim=1, keepdim=True) + self.eps).rsqrt())
