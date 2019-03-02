import torch
import torch.nn as nn
from torch.nn import utils
from torchvision import models
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
            # i_c = min(mult*ngf, 512)
            # o_c = min(2*mult*ngf, 512)
            i_c = mult*ngf
            o_c = 2*mult*ngf
            setattr(self, 'down'+str(i), Conv2dBlock_CBN(n_class, i_c, o_c, 3, 2, 1, activation))

        ### resnet block ###
        # mult = min(2**n_down, 8)
        mult = 2**n_down
        for i in range(n_blocks):
            setattr(self, 'block'+str(i), ResBlock_CBN(n_class, mult*ngf, activation, 'reflect'))

        ### up sampling ###
        for i in range(n_down):
            mult = 2**(n_down-i)
            # i_c = min(mult*ngf, 512)
            # o_c = min(mult*ngf//2, 512)
            i_c = mult*ngf
            o_c = mult*ngf//2
            # setattr(self, 'up'+str(i), trConv2dBlock_CBN(n_class, i_c, o_c, 2, 2, 0, activation))
            setattr(self, 'up'+str(i), upConv2dBlock_CBN(n_class, i_c, o_c, 3, 1, 1, activation))

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

class ResNetGenerator_CBN2(nn.Module):
    def __init__(self, n_class, input_nc, output_nc, ngf, n_down, n_blocks, activation):
        super(ResNetGenerator_CBN2, self).__init__()

        self.n_down = n_down
        self.n_blocks = n_blocks

        self.input_conv = Conv2dBlock(input_nc, ngf, 7, 1, 3, 'none', activation, 'reflect')

        ### down sampling ###
        for i in range(n_down):
            mult = 2**i
            # i_c = min(mult*ngf, 512)
            # o_c = min(2*mult*ngf, 512)
            i_c = mult*ngf
            o_c = 2*mult*ngf
            setattr(self, 'down'+str(i), Conv2dBlock(i_c, o_c, 3, 2, 1, 'instance', activation, 'reflect'))

        ### resnet block ###
        # mult = min(2**n_down, 8)
        mult = 2**n_down
        for i in range(n_blocks):
            setattr(self, 'block'+str(i), ResBlock_CBN(n_class, mult*ngf, activation, 'reflect'))

        ### up sampling ###
        for i in range(n_down):
            mult = 2**(n_down-i)
            # i_c = min(mult*ngf, 512)
            # o_c = min(mult*ngf//2, 512)
            i_c = mult*ngf
            o_c = mult*ngf//2
            # setattr(self, 'up'+str(i), trConv2dBlock_CBN(n_class, i_c, o_c, 2, 2, 0, activation))
            setattr(self, 'up'+str(i), upConv2dBlock(i_c, o_c, 3, 1, 1, 'instance', activation, 'reflect'))

        self.output_conv = Conv2dBlock(ngf, output_nc, 7, 1, 3, 'none', 'tanh', 'reflect')

    def forward(self, x, c):
        x = self.input_conv(x)

        for i in range(self.n_down):
            conv = getattr(self, 'down'+str(i))
            x = conv(x)

        for i in range(self.n_blocks):
            conv = getattr(self, 'block'+str(i))
            x = conv(x, c)

        for i in range(self.n_down):
            conv = getattr(self, 'up'+str(i))
            x = conv(x)

        x = self.output_conv(x)
        return x


class ProjectionDiscriminator(nn.Module):
    def __init__(self, n_class, input_nc, ndf, num_D, n_layer, norm, activation):
        super(ProjectionDiscriminator, self).__init__()

        self.input_nc = input_nc
        self.ndf = ndf
        self.n_layer = n_layer
        self.norm = norm
        self.activation = activation
        c = 2**(n_layer)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.models = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.embeds = nn.ModuleList()
        for i in range(num_D):
            self.models.append(self.make_net())
            self.linears.append(LinearBlock(c*ndf, 1, 'sn', 'none'))
            self.embeds.append(utils.spectral_norm(nn.Embedding(n_class, c*ndf)))

    def make_net(self):
        model = [Conv2dBlock(self.input_nc, self.ndf, 4, 2, 2, self.norm, self.activation)]
        for n in range(self.n_layer):
            mult = 2**n
            i_c = min(mult*self.ndf, 512)
            o_c = min(2*mult*self.ndf, 512)
            model += [Conv2dBlock(i_c, o_c, 4, 2, 2, self.norm, self.activation)]
        return nn.Sequential(*model)

    def forward(self, x, c):
        outputs = []
        for model, linear, embed in zip(self.models, self.linears, self.embeds):
            h = model(x)
            h = torch.sum(h, dim=(2, 3))
            output = linear(h)
            output += torch.sum(embed(c)*h, dim=1, keepdim=True)
            outputs.append(output)
            x = self.downsample(x)
        return outputs

################################################################################
# loss
################################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor

        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = self.real_label_var.requires_grad_(False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                self.fake_label_var = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = self.fake_label_var.requires_grad_(False)
            target_tensor = self.fake_label_var
        return target_tensor

    def forward(self, input, target_is_real):
        if isinstance(input, list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
