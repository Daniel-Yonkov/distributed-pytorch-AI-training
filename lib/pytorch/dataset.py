import os
from random import Random

import torch.distributed as dist
from torch import utils
from torchvision import datasets, transforms


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1]):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(os.environ['RANDOM_SEED'])
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition) -> Partition:
        return Partition(self.data, self.partitions[partition])


def partition_dataset() -> tuple([utils.data.DataLoader, int]):
    """ Partitioning MNIST """
    dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    size = dist.get_world_size()
    bsz = 128 / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = utils.data.DataLoader(
        partition, batch_size=int(bsz), shuffle=True)
    return train_set, bsz


def testDataset():
    return datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
