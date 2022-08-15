import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.autograd import Variable
from math import ceil
from lib.pytorch.NeuralNetwork import Net


def trainModel(trainSet, batchSize, numberOfEpochs, rank):
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    num_batches = ceil(len(trainSet.dataset) / float(batchSize))
    for epoch in range(numberOfEpochs):
        epoch_loss = 0.0
        for data, target in trainSet:
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.data.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches)
    return model


def test(model, test_dataset):
    model.eval()
    test_loss = 0
    correct = 0
    test_loader = torch.utils.data.DataLoader(test_dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
