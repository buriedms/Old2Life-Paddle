import paddle.io
import random
from data.base_data_loader import BaseDataLoader
from data import online_dataset_for_old_photos as dts_ray_bigfile


def CreateDataset(opt):
    dataset = None
    if opt.training_dataset == 'domain_A' or opt.training_dataset == 'domain_B':
        dataset = dts_ray_bigfile.UnPairOldPhotos_SR()
    if opt.training_dataset == 'mapping':
        if opt.random_hole:
            dataset = dts_ray_bigfile.PairOldPhotos_with_hole()
        else:
            dataset = dts_ray_bigfile.PairOldPhotos()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        # print(self.dataset.__len__())
        # print(self.dataset.__getitem__(5))
        # raise NotImplementedError
        self.dataloader = paddle.io.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads),
            drop_last=True)
        if opt.isTrain and len(opt.gpu_ids) > 1:
            self.batchsampler = paddle.io.DistributedBatchSampler(dataset=self.dataset, batch_size=opt.batchSize,
                                                                  shuffle=not opt.serial_batches, drop_last=True)
            self.dataloader = paddle.io.DataLoader(
                self.dataset,
                batch_sampler=self.batchsampler,
                num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
