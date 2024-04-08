from __future__ import absolute_import
import math
import torch.nn as nn
from .channel_selection import channel_selection
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ['resnet8x4']

"""
preactivation resnet with BasicBlock design.
"""

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], planes * self.expansion, kernel_size=3,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out
    
class resnet(nn.Module):
    def __init__(self, depth, dataset='cifar100', cfg=None):
        super(resnet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 9n+2'

        n = (depth - 2) // 6

        if cfg == None:
            cfg = [[32, 64], (n-1)*[64, 64], [64, 128], (n-1)*[128,128], [128, 256], (n-1)*[256, 256], [256]]
            cfg = [item for sub_list in cfg for item in sub_list]

        block = BasicBlock
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 64, n, cfg[0:2*n])
        self.layer2 = self._make_layer(block, 128, n, cfg[2*n:4*n], stride=2)
        self.layer3 = self._make_layer(block, 256, n, cfg[4*n:6*n], stride=2)
        self.bn = nn.BatchNorm2d(256 * block.expansion)
        self.select = channel_selection(256 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        if dataset == 'cifar10':
            self.fc = nn.Linear(cfg[-1], 10)
        elif dataset == 'cifar100':
            self.fc = nn.Linear(cfg[-1], 100)

        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:2], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[2*i:2*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet8x4(depth, dataset='cifar100', cfg=None):
    return resnet(depth=depth, dataset=dataset, cfg=cfg)


def resnet32x4(depth, dataset='cifar100'):
    return resnet(depth=depth, dataset=dataset)

if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 32, 32)
    net = resnet(depth=8)
    print(net)
    logits = net(x)
    print(logits.size())
