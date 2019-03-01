import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=opt.batch_size, shuffle=not opt.no_shuffle)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
