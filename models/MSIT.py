import torch
from .base_model import BaseModel
from . import networks


class MSIT(BaseModel):
    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.n_class = opt.n_weather_class
        # self.n_class = opt.n_timeofday_class

        # self.loss_names = ['gen', 'dis', 'adv', 'rec']
        self.loss_names = ['gen', 'dis', 'adv', 'vgg']
        self.model_names = ['gen']
        # self.gen = networks.ResNetGenerator_CBN(self.n_class, opt.input_nc, opt.output_nc, opt.ngf, opt.n_down, opt.n_blocks, 'relu')
        # self.gen = networks.ResNetGenerator_CBN2(self.n_class, opt.input_nc, opt.output_nc, opt.ngf, opt.n_down, opt.n_blocks, 'relu')
        # self.gen = networks.ResNetGenerator_CBN3(self.n_class, opt.input_nc, opt.output_nc, opt.ngf, opt.n_down, opt.n_blocks, 'relu')
        # self.gen = networks.ResNetGenerator_AdaIN(self.n_class, opt.input_nc, opt.output_nc, opt.ngf, opt.n_down, opt.n_blocks)
        self.gen = networks.cResNetGenerator(self.n_class, opt.input_nc, opt.output_nc, opt.ngf, opt.n_down, opt.n_blocks, 'relu')

        if self.isTrain:
            ### define Discriminator ###
            self.model_names += ['dis']
            dis_input_nc = opt.output_nc + opt.label_nc
            self.dis = networks.Discriminator(dis_input_nc, opt.ndf, opt.num_D, opt.n_layer)
            # self.dis = networks.DiscriminatorSA(dis_input_nc, opt.ndf, opt.num_D, opt.n_layer)
            # self.dis = networks.ProjectionDiscriminator(self.n_class, dis_input_nc, opt.ndf, opt.num_D, opt.n_layer, 'sn', 'lrelu')

            ### set optimizers ###
            self.optimizer_names = ['optimizer_G', 'optimizer_D']
            self.optimizer_G = torch.optim.Adam(self.gen.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizer_D = torch.optim.Adam(self.dis.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            # self.optimizer_G = torch.optim.Adam([p for p in self.gen.parameters() if p.requires_grad],
            #                                     lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            # self.optimizer_D = torch.optim.Adam([p for p in self.dis.parameters() if p.requires_grad],
            #                                     lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)

            ### set loss functions ###
            self.criterionGAN = networks.GANLoss()
            self.criterionVGG = networks.VGGLoss()

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def set_variables(self, data):
        self.image = data['image'].cuda()
        self.label = data['label'].cuda()
        self.category = data['weather'].cuda()
        # self.category = data['timeofday'].cuda()

    def update_D(self):
        self.set_requires_grad([self.dis], True)
        self.optimizer_D.zero_grad()

        fake_image = self.gen(self.label, self.category)
        fake_pair = torch.cat((fake_image.detach(), self.label), dim=1)
        real_pair = torch.cat((self.image, self.label), dim=1)

        # c_onehot = torch.cuda.FloatTensor(c.size(0), self.n_class).zero_()
        # c = c.unsqueeze(1)
        # c_onehot.scatter_(1, c, 1)
        # c_onehot = c_onehot.unsqueeze(-1).unsqueeze(-1)
        # c_onehot = c_onehot.repeat(1, 1, self.image.size(2), self.image.size(3))
        # fake_pair = torch.cat((fake_image.detach(), self.label, c_onehot), dim=1)
        # real_pair = torch.cat((self.image, self.label, c_onehot), dim=1)

        # D_fake = self.dis(fake_pair, self.category)
        # D_real = self.dis(real_pair, self.category)
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

        self.fake_image = self.gen(self.label, self.category)
        fake_pair = torch.cat((self.fake_image, self.label), dim=1)

        # D_fake = self.dis(fake_pair, self.category)
        D_fake = self.dis(fake_pair)
        self.loss_adv = self.criterionGAN(D_fake, True)
        self.loss_vgg = self.criterionVGG(self.fake_image, self.image) * self.opt.lambda_vgg

        self.loss_gen = self.loss_adv + self.loss_vgg
        self.loss_gen.backward()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.update_D()
        self.update_G()

    def forward(self):
        fake_images = []
        with torch.no_grad():
            for c in range(self.n_class):
                fake_image = self.gen(self.label, torch.tensor([c]).long().cuda())
                fake_images.append(fake_image.cpu())
        fake_images = torch.cat(fake_images, dim=0)
        return fake_images
