from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader



opt = TrainOptions().parse()

### set dataloader ###
print('### prepare DataLoader ###')
data_loader = CreateDataLoader(opt)
train_loader = data_loader.load_data()
print('training images : {}'.format(len(data_loader)))
print('numof_iteration : {}'.format(len(train_loader)))
