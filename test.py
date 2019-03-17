from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.create_model import create_model
from utils import visualizer


opt = TestOptions().parse()

### set dataloader ###
print('### prepare DataLoader ###')
data_loader = CreateDataLoader(opt)
test_loader = data_loader.load_data()
print('training images : {}'.format(len(data_loader)))
print('numof_iteration : {}'.format(len(test_loader)))

### define model ###
model = create_model(opt)
model.gen.eval()

print('### start test ! ###')
for iter, data in enumerate(test_loader):
    model.set_variables(data)
    fake_image = model.forward()

    if isinstance(fake_image, list):
        fake_image = torch.cat(fake_image, dim=0)

    fake_image = (fake_image + 1.0) / 2.0
    visualizer.save_test_images(opt, iter, data['label'], fake_image)
