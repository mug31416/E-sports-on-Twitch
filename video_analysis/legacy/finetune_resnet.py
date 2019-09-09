#!/usr/bin/env python3
import torch
import torch.nn as nn
from torchvision.models import resnet18
from pytorch_data import *


class LinearND(nn.Module):

  def __init__(self, in_features, out_features, bias=True):
    """
    A torch.nn.Linear layer modified to accept batch x time x feature tensors.
    The function treats the last dimension of the input as the feature/hidden layer dimension.
    It uses the Linear module that has a nice Xavier weight initialization.
    """
    super(LinearND, self).__init__()
    self.fc = nn.Linear(in_features, out_features, bias)
    my_initializer(self.fc)

  def forward(self, x):
    dims = x.size()
    out = x.contiguous().view(-1, dims[-1])
    out = self.fc(out)
    dims = list(dims)
    dims[-1] = out.size()[-1]
    return out.view(tuple(dims))

# Resnet18/34 layers
# (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (relu): ReLU(inplace)
# (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
# (layer1): Sequential(...)
# (layer2): Sequential(...)
# (layer3): Sequential(...)
# (layer4): Sequential(...)
# (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
# (fc): Linear(in_features=512, out_features=1000, bias=True)
class ResNetExtractor(nn.Module):
  def __init__(self, feat_dim, out_dim, drop_out, freeze_old=True):
    super(ResNetExtractor, self).__init__()
    print(f'Feature dim {feat_dim} output dim {out_dim} freeze layers {freeze_old}')
    pmod = resnet18(pretrained=True)
    self.frozen = nn.Sequential(pmod.conv1, pmod.bn1, pmod.relu, pmod.maxpool,
                               pmod.layer1, pmod.layer2, pmod.layer3, pmod.layer4,
                               pmod.avgpool)

    self.change_freeze(freeze_old)

    self.fc_feat = LinearND(in_features=512, out_features=feat_dim, bias=True)
    self.fc_relu = nn.ReLU()
    self.drop_out = nn.Dropout(drop_out)
    self.fc_out = LinearND(in_features=feat_dim, out_features=out_dim, bias=True)

  def change_freeze(self, freeze_old):

    if freeze_old:
      print('Using frozen model!')
      requires_grad = False
    else:
      print('Unfreeze the original model')
      requires_grad = True

    for child in self.frozen.children():
      for param in child.parameters():
        param.requires_grad = requires_grad

  def reset_dropout(self, drop_out):
    self.drop_out = nn.Dropout(drop_out)

  def get_feat(self, x):

    xfr = self.frozen(x)
    feat = self.fc_feat(xfr.squeeze())

    return self.fc_relu(self.drop_out(feat))
                                
  def forward(self, x):

    feat = self.get_feat(x)
    return self.fc_out(feat)

