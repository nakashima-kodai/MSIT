from options.base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        ''' for display '''
        self.parser.add_argument('--save_epoch_freq', type=int, default=10)
        self.parser.add_argument('--print_freq', type=int, default=100)

        ''' for training '''
        self.parser.add_argument('--epoch', type=int, default=100)
        self.parser.add_argument('--continue_train', action='store_true')
        self.parser.add_argument('--load_epoch', type=int, default=0)
        self.parser.add_argument('--load_pretrain', type=str, default='')
        self.parser.add_argument('--phase', default='train')
        self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--niter', type=int, default=100, help='# of epoch at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')

        ''' for discriminator '''
        self.parser.add_argument('--netD', type=str, default='multi_patch')
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64)

        ''' for losses '''
        self.parser.add_argument('--lambda_feat', type=float, default=10.0)
        self.parser.add_argument('--no_ganFeat_loss', action='store_true')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        self.parser.add_argument('--no_lsgan', action='store_true')

        self.isTrain = True
