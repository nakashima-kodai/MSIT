from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from utils import visualizer


opt = TrainOptions().parse()

### set dataloader ###
print('### prepare DataLoader ###')
data_loader = CreateDataLoader(opt)
train_loader = data_loader.load_data()
print('training images : {}'.format(len(data_loader)))
print('numof_iteration : {}'.format(len(train_loader)))


for iter, data in enumerate(train_loader):
    image = (data['image'] + 1.0) / 2.0
    visualizer.show_image(data['label'], image, nrow=opt.batch_size)
