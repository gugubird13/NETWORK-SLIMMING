from __future__ import absolute_import
from .channel_selection import channel_selection

'''preresnet for cifar dataset.
为剪枝准备
'''
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['resnet']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None, is_last=False):
        super(BasicBlock, self).__init__()

        self.is_last = is_last
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = conv3x3(cfg[0], cfg[1], stride)
        
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = conv3x3(cfg[1], planes)     
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        for param in self.select.parameters():
            param.requires_grad = False

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class ResNet(nn.Module):

    def __init__(self, depth, num_filters, block_name='BasicBlock', num_classes=100, cfg=None):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        # 深度就是网络的深度，n就是重复使用多少次block了，basicblock或者bottleneck
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        else:
            raise ValueError('block_name shoule be Basicblock')

        if cfg == None:
            cfg = [[32, 64], (n-1)*[64, 64], [64, 128], (n-1)*[128,128], [128, 256], (n-1)*[256, 256], [256]]
            cfg = [item for sub_list in cfg for item in sub_list]

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
        self.fc = nn.Linear(cfg[-1] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for param in self.select.parameters():
            param.requires_grad = False

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

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.relu)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')

        return [bn1, bn2, bn3]

    def forward(self, x, is_feat=False, preact=False):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet8(**kwargs):
    return ResNet(8, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet14(**kwargs):
    return ResNet(14, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet20(**kwargs):
    return ResNet(20, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet32(**kwargs):
    return ResNet(32, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet44(**kwargs):
    return ResNet(44, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet56(**kwargs):
    return ResNet(56, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet110(**kwargs):
    return ResNet(110, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet8x4(**kwargs):
    return ResNet(8, [32, 64, 128, 256], 'basicblock', **kwargs)


def resnet32x4(**kwargs):
    return ResNet(32, [32, 64, 128, 256], 'basicblock', **kwargs)


if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 32, 32)
    checkpoint = torch.load("/home/szy/network-slimming/experiments/prune_cifar_resnet/pruned.pth.tar")
    model = resnet8x4(num_classes=100,cfg=checkpoint['cfg'])
    model.load_state_dict(checkpoint['state_dict'])
    logits = model(x)
    print(logits.size())
    for i, (name, param) in enumerate(model.named_parameters()):
        if i in [3,11,19,27]:
            print(f"Index: {i}, Name: {name}, Param: {param}")
    for m in model.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')
        else:
            print('warning')
