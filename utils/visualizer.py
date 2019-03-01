import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision


def show_loaded_image(label_tensor, truth_tensor):
    batch_size = truth_tensor.shape[0]

    labeltruth = torch.cat((label_tensor[0].unsqueeze(0), truth_tensor[0].unsqueeze(0)), dim=0)
    for i in range(1, batch_size):
        temp = torch.cat((label_tensor[i].unsqueeze(0), truth_tensor[i].unsqueeze(0)), dim=0)
        labeltruth = torch.cat((labeltruth, temp), dim=0)

    labeltruth = torchvision.utils.make_grid(labeltruth, nrow=6, padding=1)
    labeltruth = labeltruth.numpy().transpose((1, 2, 0))

    plt.imshow(labeltruth)

def save_images(opt, epoch, label, truth, fake_image):
    batch_size = label.shape[0]
    lbltrufake = torch.cat((label[0].unsqueeze(0), truth[0].unsqueeze(0), fake_image[0].unsqueeze(0)), dim=0)
    for i in range(1, opt.batch_size):
        temp = torch.cat((label[i].unsqueeze(0), truth[i].unsqueeze(0), fake_image[i].unsqueeze(0)), dim=0)
        lbltrufake = torch.cat((lbltrufake, temp), dim=0)

    lbltrufake = torchvision.utils.make_grid(lbltrufake, nrow=3*4, padding=1)
    lbltrufake = lbltrufake.numpy().transpose((1, 2, 0))

    image_name = str(epoch).zfill(3) + '.png'
    save_path = os.path.join(opt.sample_dir, opt.name, image_name)
    print('save_path: {}'.format(save_path))
    plt.imsave(save_path, lbltrufake)
