"""
  FileName     [ cnn.py ]
  PackageName  [ HW4 ]
  Synopsis     [ ResNet models for feature extracting ]

  * Feature extracting models:
    Resnet (Resnet18, Resnet34, Resnet50, Resnet101, Resnet152)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torch.autograd import Function
from torch.utils import model_zoo as model_zoo
from torch.utils.data import DataLoader
from torchvision import transforms

import torchsummary
# import dataset
import utils

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

DEVICE = utils.selectDevice()

class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, groups=1, width_per_group=64,
                replace_stride_with_dilation=None):
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        self.inplanes = 64
        self.dilation = 1
        self.groups   = groups
        self.base_width = width_per_group

        super(ResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn1   = nn.BatchNorm2d(self.inplanes)
        self.relu  = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=(8, 10), stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # ------------------------------------------------------------------------------------------------
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # ------------------------------------------------------------------------------------------------
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    
    def make_layer(self, block: nn.Module, planes, num_blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, groups=self.groups, base_width=self.base_width, dilation=previous_dilation))
        self.inplanes = planes * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        return x.view(x.shape[0], -1)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, activation=nn.ReLU(inplace=True), 
                downsample=None, groups=1, base_width=64, dilation=1):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64. )) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(width)
        
        self.conv3 = nn.Conv2d(width, width * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(width * self.expansion)

        self.activation = activation
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):        
        identical = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identical = self.downsample(identical)

        out = self.activation(out + identical)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, activation=nn.ReLU(inplace=True), groups=1,
                downsample=None, base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.activation = activation
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identical = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identical = self.downsample(identical)

        out = self.activation(out + identical)

        return out

def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, (2, 2, 2, 2))

    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet18'])
        del state_dict['fc.weight'], state_dict['fc.bias']
        model.load_state_dict(state_dict)

    return model 

def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, (3, 4, 6, 3))

    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet34'])
        del state_dict['fc.weight'], state_dict['fc.bias']
        model.load_state_dict(state_dict)
        
    return model

def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, (3, 4, 6, 3))

    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        del state_dict['fc.weight'], state_dict['fc.bias']
        model.load_state_dict(state_dict)

    return model

def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, (3, 4, 23, 3))

    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet101'])
        del state_dict['fc.weight'], state_dict['fc.bias']
        model.load_state_dict(state_dict)

    return model

def resnet152(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, (3, 8, 36, 3))

    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet152'])
        del state_dict['fc.weight'], state_dict['fc.bias']
        model.load_state_dict(state_dict)

    return model

def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()
