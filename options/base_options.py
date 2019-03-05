import os
import argparse
import torch
from utils.utils import mkdir


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        ### experiment specifics ###
        self.parser.add_argument('--model', type=str, default='MSIT')
        self.parser.add_argument('--name', type=str, default='label2image')
        self.parser.add_argument('--gpu_ids', type=str, default='0')
        self.parser.add_argument('--ckpt_dir', type=str, default='./ckpt')
        self.parser.add_argument('--sample_dir', type=str, default='./samples')
        self.parser.add_argument('--result_dir', type=str, default='./results')

        ### input/output sizes ###
        self.parser.add_argument('--batch_size', type=int, default=64)
        self.parser.add_argument('--load_size', type=int, default=64)
        self.parser.add_argument('--crop_size', type=int, default=64)
        self.parser.add_argument('--label_nc', type=int, default=3)
        self.parser.add_argument('--input_nc', type=int, default=3)
        self.parser.add_argument('--output_nc', type=int, default=3)

        ### input label ###
        self.parser.add_argument('--n_weather_class', type=int, default=6)
        self.parser.add_argument('--n_timeofday_class', type=int, default=3)

        ### for setting inputs ###
        self.parser.add_argument('--data_root', type=str, default='../datasets/bdd100k')
        self.parser.add_argument('--scale_transform', type=str, default='resize', help='scale images to this size at load time [resize, resize_and_crop, crop, scale_width, scale_width_and_crop]')
        self.parser.add_argument('--no_shuffle', action='store_true')
        self.parser.add_argument('--no_flip', action='store_true')

        ### for generator and enhancer ###
        self.parser.add_argument('--ngf', type=int, default=64)
        # for generator #
        self.parser.add_argument('--n_down', type=int, default=3)
        self.parser.add_argument('--n_blocks', type=int, default=3)
        # for enhancer #
        self.parser.add_argument('--n_enhancers', type=int, default=1)
        self.parser.add_argument('--n_blocks_local', type=int, default=3)
        self.parser.add_argument('--n_epoch_fix_local', type=int, default=0)

        ### for generator/discriminator ###
        self.parser.add_argument('--init_type', type=str, default='normal', help='[normal, xavier, kaiming]')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain

        ### print options to console ###
        args = vars(self.opt)
        print('--------------- Options ---------------')
        for k, v in sorted(args.items()):
            print('{:<21} : {}'.format(str(k), str(v)))
        print('----------------- End -----------------\n')

        ### make directory ###
        if self.isTrain:
            expr_dir = os.path.join(self.opt.ckpt_dir, self.opt.name)
            smpl_dir = os.path.join(self.opt.sample_dir, self.opt.name)
            mkdir(expr_dir)
            mkdir(smpl_dir)

            ### save options to the disk ###
            file_name = os.path.join(expr_dir, 'options.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('{:<21} : {}\n'.format(str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        else:
            rslt_dir = os.path.join(self.opt.result_dir, self.opt.name)
            mkdir(rslt_dir)

        ### set CUDA device ###
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
            torch.backends.cudnn.benchmark=True

        return self.opt
