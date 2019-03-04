import os
import torch
from collections import OrderedDict
from . import utils


class BaseModel():
    def initialize(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.ckpt_dir, opt.name)

        self.model_names = []
        self.loss_names = []
        self.optimizer_names = []

    def setup(self):
        if self.isTrain:
            self.schedulers = self.get_schedulers()

        if not self.isTrain:
            self.load_networks(self.opt.load_epoch)
        else:
            self.init_networks()

        if len(self.opt.gpu_ids):
            self.to_cuda()

    def init_networks(self):
        for name in self.model_names:
            net = getattr(self, name)
            utils.init_weights(net, self.opt.init_type)

    def get_schedulers(self):
        schedulers = []
        for name in self.optimizer_names:
            optimizer = getattr(self, name)
            schedulers.append(utils.get_scheduler(optimizer, self.opt))
        return schedulers

    def to_cuda(self):
        for name in self.model_names:
            net = getattr(self, name)
            net = torch.nn.DataParallel(net.cuda(), device_ids=self.opt.gpu_ids)
            setattr(self, name, net)

    def update_lr(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = getattr(self, self.optimizer_names[0]).param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def save_networks(self, epoch):
        for name in self.model_names:
            save_filename = '%s_%s.pth' % (epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, name)

            print('saving the {} to {}'.format(name, save_path))
            state_dict = net.module.cpu().state_dict()
            torch.save(state_dict, save_path)
            net.cuda()

    # load models from the disk
    def load_networks(self, epoch, save_dir=None):
        for name in self.model_names:
            load_filename = '%s_%s.pth' % (epoch, name)
            if save_dir is None:
                save_dir = self.save_dir
            load_path = os.path.join(save_dir, load_filename)
            net = getattr(self, name)

            print('loading the {} from {}'.format(name, load_path))
            state_dict = torch.load(load_path)
            net.load_state_dict(state_dict)

    def print_networks(self):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            # float(...) works for both scalar tensor and float number
            errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
