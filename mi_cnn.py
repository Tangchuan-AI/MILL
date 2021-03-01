# coding:utf8

# An implementation of MIL_CNN.
# "Deep Multiple Instance Convolutional Neural Networks for Learning Robust Scene Representations"

from .basic_module import BasicModule
from torch import nn
from torch.nn import functional as F
import torch
from config import opt
import torchvision

class MI_CNN(BasicModule):

    def __init__(self, num_classes=2):
        super(MI_CNN, self).__init__()
        self.model_name = 'mi_cnn'

        pretrained_net = torchvision.models.vgg16(pretrained=True, progress=True)
        self.backbone = nn.Sequential(*list(pretrained_net.children())[:-2])
        self.backbone_channel = 512
        self.outchannel = 128

        self.conv1 = nn.Conv2d(self.backbone_channel, 512, 1, padding=0, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(self.backbone_channel, 512, 3, padding=1, dilation=1, bias=False)
        self.conv3 = nn.Conv2d(self.backbone_channel, 512, 3, padding=2, dilation=2, bias=False)
        self.conv4 = nn.Conv2d(self.backbone_channel, 512, 3, padding=3, dilation=3, bias=False)

        self.conv5 = nn.Conv2d(512, 256, 1, padding=0, dilation=1, bias=False)
        self.conv6 = nn.Conv2d(256, 2, 1, padding=0, dilation=1, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        x = x1 + x2 + x3 + x4

        x5 = self.conv5(x)
        x6 = self.conv6(x5)

        y = self.mil_pooling(x5, x6)
        
        return y

    def mil_pooling(self, X, S):
        (nbatches, nchannels, width, hight) = X.size()
        nclasses = S.size(-3)
        W = torch.rand(nchannels, requires_grad=True).to(opt.device)
        b = torch.zeros(1, requires_grad=True).to(opt.device)
        Y = torch.zeros((nbatches, nclasses), requires_grad=True).to(opt.device)
        
        W = W.unsqueeze(-1)
        W = W.unsqueeze(-1)
        W = W.unsqueeze(0)
        a = torch.exp(torch.sigmoid(torch.sum(W*X + b, dim=-3)))
        a = a.unsqueeze(-3)
        Y = torch.sum(S * a, dim=[-1,-2]) / torch.sum(a, dim=[-1,-2])
    
        return Y