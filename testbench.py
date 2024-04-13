from torchvision import datasets, transforms
from models import *
from torch.autograd import Variable
import torch.nn.functional as F

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset)), {'test_loss': test_loss, 'top1': 100. * correct / len(test_loader.dataset)} 

if __name__ == '__main__':
    import torch

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('/home/szy/DIST_KD/classification/data/cifar/cifar-100-python', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=128, shuffle=True)

    checkpoint = torch.load("/home/szy/network-slimming/experiments/sparisty/cifar8x4_with_dist_v2/model_best.pth.tar")
    model = resnet8x4()
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    test()