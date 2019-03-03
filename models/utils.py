import functools
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from . import normalizations


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        net = torch.nn.DataParallel(net.cuda(), gpu_ids)
    init_weights(net, init_type)
    return net

def get_pad_layer(padding=0, pad_type='zero'):
    if pad_type == 'reflect':
        pad_layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        pad_layer = nn.ReplicationPad2d(padding)
    elif pad_type == 'zero':
        pad_layer = nn.ZeroPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{}] is not found'.format(pad_type))

    return pad_layer

def get_norm_layer(norm_nc, norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d(norm_nc)
    elif norm_type == 'instance':
        # norm_layer = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
        norm_layer = nn.InstanceNorm2d(norm_nc)
    elif norm_type == 'adain':
        norm_layer = normalizations.AdaptiveInstanceNorm2d(norm_nc)
    elif norm_type == 'ln':
        norm_layer = normalizations.LayerNorm(norm_nc)
    elif norm_type == 'none' or norm_type == 'sn':
        norm_layer = None
    else:
        raise NotImplementedError('norm layer [{}] is not found'.format(norm_type))
    return norm_layer

def get_activation_layer(activation_type='relu'):
    if activation_type == 'relu':
        activation = nn.ReLU(inplace=True)
    elif activation_type == 'lrelu':
        activation = nn.LeakyReLU(0.2, inplace=True)
    elif activation_type == 'prelu':
        activation = nn.PReLU()
    elif activation_type == 'selu':
        activation = nn.SELU(inplace=True)
    elif activation_type == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation_type == 'tanh':
        activation = nn.Tanh()
    elif activation_type == 'none':
        activation = None
    else:
        raise NotImplementedError('padding layer [{}] is not found'.format(activation_type))
    return activation

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.epoch) / float(opt.epoch_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.epoch_decay, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [{}] is not implemented'.format(opt.lr_policy))
    return scheduler
