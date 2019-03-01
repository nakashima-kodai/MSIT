import torch
import torch.nn as nn
from .blocks import *


class ResNetGenerator_CBN(nn.Module):
    def __init__(self, n_class, input_nc, output_nc, ngf, n_down, n_blocks, activation):
        super(ResNetGenerator_CBN, self).__init__()

        self.n_down = n_down
        self.n_blocks = n_blocks

        self.input_conv = Conv2dBlock_CBN(n_class, input_nc, ngf, 7, 1, 3, activation, 'reflect')

        ### down sampling ###
        for i in range(n_down):
            mult = 2**i
            setattr(self, 'down'+str(i), Conv2dBlock_CBN(n_class, ngf, mult*ngf, 3, 2, 1, activation))

        ### resnet block ###
        mult = 2**n_down
        for i in range(n_blocks):
            setattr(self, 'block'+str(i), ResBlock_CBN(n_class, mult*ngf, activation, 'reflect'))

        ### up sampling ###
        for i in range(n_down):
            mult = 2**(n_down-i)
            setattr(self, 'up'+str(i), upConv2dBlock_CBN(n_class, mult*ngf, mult*ngf//2, 5, 1, 2, activation))

        self.output_conv = Conv2dBlock(ngf, output_nc, 7, 1, 3, 'none', 'tanh', 'reflect')

    def forward(self, x, c):
        x = self.input_conv(x, c)

        for i in range(self.n_down):
            conv = getattr(self, 'down'+str(i))
            x = conv(x, c)

        for i in range(self.n_blocks):
            conv = getattr(self, 'block'+str(i))
            x = conv(x, c)

        for i in range(self.n_down):
            conv = getattr(self, 'up'+str(i))
            x = conv(x, c)

        x = self.output_conv(x)
        return x
