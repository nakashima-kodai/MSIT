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
        self.optimizers = []

    def setup(self):
        if self.isTrain:
            self.schedulers = [utils.get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]

        if not self.isTrain or self.opt.continue_train:
            self.load_networks(self.opt.load_epoch)
        else:
            self.init_networks()

        if len(self.opt.gpu_ids):
            self.to_cuda()

    def init_networks(self):
        for name in self.model_names:
            net = getattr(self, name)
            utils.init_weights(net, self.opt.init_type)

    def to_cuda(self):
        for name in self.model_names:
            net = getattr(self, name)
            net = torch.nn.DataParallel(net.cuda(), self.opt.gpu_ids)

    def update_lr(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def save_networks(self, epoch):
        for name in self.model_names:
            save_filename = '%s_%s.pth' % (epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, name)

            print('saving the model to {}'.format(save_path))
            torch.save(net.cpu().state_dict(), save_path)
            net.cuda()

    # load models from the disk
    def load_networks(self, epoch, save_dir=None):
        for name in self.model_names:
            load_filename = '%s_%s.pth' % (epoch, name)
            if save_dir is None:
                save_dir = self.save_dir
            load_path = os.path.join(save_dir, load_filename)
            net = getattr(self, name)

            print('loading the model from {}'.format(load_path))
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
