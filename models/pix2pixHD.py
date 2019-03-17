import torch
from .base_model import BaseModel
from . import networks


class pix2pixHD(BaseModel):
    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.loss_names = ['gen', 'dis', 'adv', 'vgg']
        self.model_names = ['gen']
        self.gen = networks.ResNetEnhancer(opt.input_nc, opt.output_nc, opt.ngf, opt.n_down, opt.n_blocks, opt.n_enhancers, opt.n_blocks_local)

        if self.isTrain:
            ### define Discriminator ###
            self.model_names += ['dis']
            dis_input_nc = opt.output_nc + opt.label_nc
            self.dis = networks.Discriminator(dis_input_nc, opt.ndf, opt.num_D, opt.n_layer)

            ### set optimizers ###
            self.optimizer_names = ['optimizer_G', 'optimizer_D']
            if opt.n_epoch_fix_local > 0:
                finetune_list = set()
                gen_dict = dict(self.gen.named_parameters())
                params = []
                for k, v in gen_dict.items():
                    if k.startswith('model'+str(opt.n_enhancers)):
                        params += [v]
                        finetune_list.add(k.split('.')[0])
                print('------------- Only training the local enhancer network (for {} epochs) ------------'.format(opt.n_epoch_fix_local))
                print('The layers that are finetuned are ', sorted(finetune_list))
            else:
                params = list(self.gen.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizer_D = torch.optim.Adam(self.dis.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)

            self.criterionGAN = networks.GANLoss()
            self.criterionVGG = networks.VGGLoss()

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def set_variables(self, data):
        self.image = data['image'].cuda()
        self.label = data['label'].cuda()

    def update_D(self):
        self.set_requires_grad([self.dis], True)
        self.optimizer_D.zero_grad()

        fake_image = self.gen(self.label)
        fake_pair = torch.cat((fake_image.detach(), self.label), dim=1)
        real_pair = torch.cat((self.image, self.label), dim=1)

        D_fake = self.dis(fake_pair)
        D_real = self.dis(real_pair)

        loss_D_fake = self.criterionGAN(D_fake, False)
        loss_D_real = self.criterionGAN(D_real, True)

        self.loss_dis = 0.5*(loss_D_fake + loss_D_real)
        self.loss_dis.backward()
        self.optimizer_D.step()

    def update_G(self):
        self.set_requires_grad([self.dis], False)
        self.optimizer_G.zero_grad()

        self.fake_image = self.gen(self.label)
        fake_pair = torch.cat((self.fake_image, self.label), dim=1)

        D_fake = self.dis(fake_pair)
        self.loss_adv = self.criterionGAN(D_fake, True)
        # self.loss_rec = self.recon_criterion(self.fake_image, self.image) * self.opt.lambda_rec
        self.loss_vgg = self.criterionVGG(self.fake_image, self.image) * self.opt.lambda_vgg
        self.loss_gen = self.loss_adv + self.loss_vgg
        self.loss_gen.backward()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.update_D()
        self.update_G()

    def update_params(self):
        params = list(self.gen.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999), weight_decay=self.opt.weight_decay)
        print('------------ Now also finetuning global generator -----------')
        self.get_schedulers()

    def forward(self):
        with torch.no_grad():
            fake_image = self.gen(self.label)

        return fake_image.cpu()
