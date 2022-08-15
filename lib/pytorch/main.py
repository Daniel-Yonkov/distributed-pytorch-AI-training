import time
import torch.distributed as dist
from lib.pytorch.dataset import partition_dataset
from lib.pytorch.training import trainModel


def run(rank, worldSize, epochs, backend='gloo'):
    start = time.time()
    # Distributed Synchronous SGD Example
    dist.init_process_group(backend, rank=rank, world_size=worldSize)
    trainSet, bsz = partition_dataset()
    model = trainModel(trainSet=trainSet, batchSize=bsz,
                       numberOfEpochs=epochs, rank=rank)
    end = time.time()
    print("Time to iterate trought", epochs, "epochs:", end - start)
