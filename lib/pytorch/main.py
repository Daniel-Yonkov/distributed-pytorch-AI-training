import os
import torch.distributed as dist
from lib.pytorch.dataset import partition_dataset
from lib.pytorch.training import trainModel


def run(rank, worldSize, epochs, backend='gloo'):
    # Distributed Synchronous SGD Example
    os.environ['MASTER_ADDR'] = 'server'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=worldSize)
    trainSet, bsz = partition_dataset()
    trainModel(trainSet=trainSet, batchSize=bsz,
               numberOfEpochs=epochs, rank=rank)
