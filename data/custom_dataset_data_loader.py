import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    ''' 创建数据集
        目前支持两种数据集
            Unaligned: 用于直接读取 图片文件 在测试的时候使用
            hd5f: 用于训练的时候读取 预先制作的 .h5 格式的训练数据集
    '''
    dataset = None

    if opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'hd5f':
        from data.hd5f_dataset import HD5FDataset
        dataset = HD5FDataset()

    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    ''' 外部调用该类 来获取相应的数据 加载器
    '''
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        ''' 
        首先根据 opt 中的参数 获取相应的 dataset
        然后再将 dataset 用 torch的 dataloader 封装
        '''
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
