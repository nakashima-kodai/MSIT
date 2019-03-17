import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision


def show_image(label_tensor, truth_tensor, nrow):
    batch_size = truth_tensor.shape[0]

    labeltruth = torch.cat((label_tensor, truth_tensor), dim=0)
    labeltruth = torchvision.utils.make_grid(labeltruth, nrow=nrow, padding=1)
    labeltruth = labeltruth.numpy().transpose((1, 2, 0))

    plt.imshow(labeltruth)
    plt.pause(0.5)
    plt.clf()

def save_images(opt, epoch, label, truth, fake_image):
    batch_size = label.shape[0]

    saved_image = torch.cat((label, truth, fake_image), dim=0)
    saved_image = torchvision.utils.make_grid(saved_image, nrow=batch_size, padding=1)
    saved_image = saved_image.numpy().transpose((1, 2, 0))

    image_name = str(epoch).zfill(3) + '.png'
    save_path = os.path.join(opt.sample_dir, opt.name, image_name)
    print('save_path: {}'.format(save_path))
    plt.imsave(save_path, saved_image)

def save_test_images(opt, iter, label, fake_image):
    batch_size = label.shape[0]

    saved_image = torch.cat((label, fake_image), dim=0)
    saved_image = torchvision.utils.make_grid(saved_image, nrow=batch_size, padding=1)
    saved_image = saved_image.numpy().transpose((1, 2, 0))

    image_name = str(iter).zfill(4) + '.png'
    save_path = os.path.join(opt.result_dir, opt.name, image_name)
    print('save_path: {}'.format(save_path))
    plt.imsave(save_path, saved_image)

def print_current_losses(epoch, iter, losses):
    message = '--- epoch : {:03d}, iters : {:03d} --- \n'.format(epoch, iter)
    for k, v in losses.items():
        message += ('{:>3} : {:.7f}\n'.format(k, v))

    print(message)
