from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.create_model import create_model
from utils import visualizer


opt = TrainOptions().parse()

### set dataloader ###
print('### prepare DataLoader ###')
data_loader = CreateDataLoader(opt)
train_loader = data_loader.load_data()
print('training images : {}'.format(len(data_loader)))
print('numof_iteration : {}'.format(len(train_loader)))

### define model ###
model = create_model(opt)

### training loop ###
print('### start training ! ###')
for epoch in range(opt.epoch+opt.epoch_decay+1):
    for iter, data in enumerate(train_loader):

        model.set_variables(data)
        model.optimize_parameters()

        if iter % opt.print_iter_freq == 0:
            losses = model.get_current_losses()
            visualizer.print_current_losses(epoch, iter, losses)

            image = (data['image'] + 1.0) / 2.0
            fake_image = (model.fake_image.cpu().detach() + 1.0) / 2.0
            visualizer.save_images(opt, epoch, data['label'], image, fake_image)

    if epoch % opt.save_epoch_freq == 0:
        image = (data['image'] + 1.0) / 2.0
        fake_image = (model.fake_image.cpu().detach() + 1.0) / 2.0
        visualizer.save_images(opt, epoch, data['label'], image, fake_image)

        model.save_networks(epoch)

    model.update_lr()

    if (opt.n_epoch_fix_local != 0) and (epoch == opt.n_epoch_fix_local):
        model.update_params()
