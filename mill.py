# coding:utf8
from .basic_module import BasicModule
from torch import nn
from torch.nn import functional as F
import torch
from config import opt
import torchvision

class MILL(BasicModule):
    
    def __init__(self, num_classes=2):
        super(MILL, self).__init__()
        self.model_name = 'mill'

        pretrained_net = torchvision.models.vgg19_bn(pretrained=True, progress=True)
        self.backbone = nn.Sequential(*list(pretrained_net.children())[:-2])
        self.backbone_channel = 512
        self.outchannel = 512
        self.instance_extract = InstanceExtract(self.backbone_channel, self.outchannel)
        self.mil_pooling = ChannelAttentionModule(self.outchannel)
        # 1*1 convolution
        self.fc = nn.Linear(self.outchannel, 2, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.instance_extract(x)
        x = self.mil_pooling(x)
        y = self.fc(x)
        return y

class InstanceExtract(nn.Module):
    
    def __init__(self, backbone_channel, outchannel, reduction=16):
        super(InstanceExtract, self).__init__()
        self.backbone_channel = backbone_channel
        self.outchannel = outchannel
        # 1*1 dilated convolution
        self.conv1 = nn.Conv2d(self.backbone_channel, self.outchannel, 1, padding=0, dilation=1, bias=False)
        # 3*3 dilated convolution， dilation=1
        self.conv2 = nn.Conv2d(self.backbone_channel, self.outchannel, 3, padding=1, dilation=1, bias=False)
        # 3*3 dilated convolution， dilation=2
        self.conv3 = nn.Conv2d(self.backbone_channel, self.outchannel, 3, padding=2, dilation=2, bias=False)
        # 3*3 dilated convolution， dilation=3
        self.conv4 = nn.Conv2d(self.backbone_channel, self.outchannel, 3, padding=3, dilation=3, bias=False)

        self.instance_classification = nn.Sequential(
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(3, 2, 1))
            nn.Conv2d(self.backbone_channel, self.outchannel, 1, padding=0, dilation=1, bias=False))
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        y = x1 + x2 + x3 + x4
        y = self.instance_classification(y)
        
        return y

class ChannelAttentionModule(nn.Module):
    
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        nbatches, nchannels, _, _ = x.size()
        y1 = self.avg_pool(x).view(nbatches, nchannels)
        y2 = self.fc(y1)
        return y2*y1 + y1