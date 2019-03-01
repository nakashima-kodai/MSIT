import os
import glob
import json
import numpy as np
import torch
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.data_root

        ''' input A (label maps) '''
        dir_A = '_A' if self.opt.label_nc == 0 else '_color'
        self.dir_A = os.path.join(opt.data_root, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ''' input B (real images) '''
        dir_B = '_B' if self.opt.label_nc == 0 else '_img'
        self.dir_B = os.path.join(opt.data_root, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))

        ''' input attribute '''
        if opt.isTrain and ('bdd100k' in opt.data_root):
            self.timeofday_list = ['daytime', 'dawn/dusk', 'night']
            self.weather_list = ['clear', 'partly cloudy', 'overcast', 'rainy', 'snowy', 'foggy']

            self.dir_attribute = os.path.join(opt.data_root, opt.phase + '_attribute')
            self.attribute_paths = sorted(glob.glob(os.path.join(self.dir_attribute, '*.json')))

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):
        ''' input A (label maps) '''
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')  # remove alpha value
        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        A_tensor = transform_A(A)

        ''' input B (real images) '''
        B_tensor = 0
        B_path = self.B_paths[index]
        B = Image.open(B_path).convert('RGB')
        transform_B = get_transform(self.opt, params)
        B_tensor = transform_B(B)

        ''' input attribute '''
        timeofday = 0
        weather = 0
        if self.opt.isTrain and ('bdd100k' in self.opt.data_root):
            attribute_path = self.attribute_paths[index]
            with open(attribute_path) as f:
                att = json.load(f)
            timeofday = att['attributes']['timeofday']
            weather = att['attributes']['weather']

            timeofday = self.timeofday_list.index(timeofday)
            weather = self.weather_list.index(weather)

        input_dict = {'label': A_tensor, 'image': B_tensor, 'timeofday': timeofday, 'weather': weather}
        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batch_size * self.opt.batch_size

    def name(self):
        return 'AlignedDataset'
