import torch
import torch.nn as nn
from torch.nn import utils
from .utils import *
from models.conditional_batch_normalization import CategoricalConditionalBatchNorm2d


class ResBlocks(nn.Module):
    def __init__(self, nc, n_blocks, norm='instance', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()

        model = []
        for i in range(n_blocks):
            model += [ResBlock(nc, norm, activation, pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class ResBlock(nn.Module):
    def __init__(self, nc, norm='instance', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(nc, nc, 3, 1, 1, norm, activation, pad_type)]
        model += [Conv2dBlock(nc, nc, 3, 1, 1, norm, 'none', pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + self.model(x)

class ResBlock_CBN(nn.Module):
    def __init__(self, n_class, nc, activation='relu', pad_type='zero'):
        super(ResBlock_CBN, self).__init__()

        self.conv1 = Conv2dBlock_CBN(n_class, nc, nc, 3, 1, 1, activation, pad_type)
        self.conv2 = Conv2dBlock_CBN(n_class, nc, nc, 3, 1, 1, 'none', pad_type)

    def forward(self, x, c):
        h = self.conv1(x, c)
        return x + self.conv2(h, c)

class Conv2dBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride=1, padding=0,
                 norm='none', activation='relu', pad_type='zero', bias=True):

        super(Conv2dBlock, self).__init__()

        self.pad = get_pad_layer(padding, pad_type)
        self.norm = get_norm_layer(output_nc, norm)
        self.activation = get_activation_layer(activation)

        if norm == 'sn':
            self.conv = utils.spectral_norm(nn.Conv2d(input_nc, output_nc, kernel_size, stride, bias=bias))
        else:
            self.conv = nn.Conv2d(input_nc, output_nc, kernel_size, stride, bias=bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class upConv2dBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride=1, padding=0,
                 norm='none', activation='relu', pad_type='zero', bias=True):
        super(upConv2dBlock, self).__init__()

        self.pad = get_pad_layer(padding, pad_type)
        self.norm = get_norm_layer(output_nc, norm)
        self.activation = get_activation_layer(activation)

        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size, stride, bias=bias)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class Conv2dBlock_CBN(nn.Module):
    def __init__(self, n_class, input_nc, output_nc, kernel_size, stride=1, padding=0,
                 activation='relu', pad_type='zero', bias=True):
        super(Conv2dBlock_CBN, self).__init__()

        self.pad = get_pad_layer(padding, pad_type)
        self.norm = CategoricalConditionalBatchNorm2d(n_class, output_nc, affine=True)
        self.activation = get_activation_layer(activation)

        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size, stride, bias=bias)

    def forward(self, x, c):
        x = self.conv(self.pad(x))
        x = self.norm(x, c)
        if self.activation:
            x = self.activation(x)
        return x

class upConv2dBlock_CBN(nn.Module):
    def __init__(self, n_class, input_nc, output_nc, kernel_size, stride=1, padding=0,
                 activation='relu', pad_type='zero', bias=True):
        super(upConv2dBlock_CBN, self).__init__()

        self.pad = get_pad_layer(padding, pad_type)
        self.norm = CategoricalConditionalBatchNorm2d(n_class, output_nc, affine=True)
        self.activation = get_activation_layer(activation)

        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size, stride, bias=bias)

    def forward(self, x, c):
        x = self.up(x)
        x = self.conv(self.pad(x))
        x = self.norm(x, c)
        if self.activation:
            x = self.activation(x)
        return x

class trConv2dBlock_CBN(nn.Module):
    def __init__(self, n_class, input_nc, output_nc, kernel_size, stride=1, padding=0,
                 activation='relu', pad_type='zero', bias=True):
        super(trConv2dBlock_CBN, self).__init__()

        self.pad = get_pad_layer(padding, pad_type)
        self.norm = CategoricalConditionalBatchNorm2d(n_class, output_nc, affine=True)
        self.activation = get_activation_layer(activation)

        self.conv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size, stride)

    def forward(self, x, c):
        x = self.conv(self.pad(x))
        x = self.norm(x, c)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm='none', activation='relu', bias=True):
        super(LinearBlock, self).__init__()

        if norm == 'batch':
            self.norm = nn.BatchNorm1d(output_nc)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm1d(output_nc)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            raise NotImplementedError('norm layer [{}] is not found'.format(norm_type))

        self.activation = get_activation_layer(activation)

        if norm == 'sn':
            self.fc = utils.spectral_norm(nn.Linear(input_nc, output_nc, bias=bias))
        else:
            self.fc = nn.Linear(input_nc, output_nc, bias=bias)

    def forward(self, x):
        x = self.fc(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
