import torch.nn as nn
import math
import numpy as np
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable

USE_MLP=False

#Code from (with tiny modifications) https://github.com/jeffreyhuang1/two-stream-action-recognition

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

def my_initializer(m):

  if hasattr(m, 'weight'):
    torch.nn.init.xavier_uniform_(m.weight.data)
  if hasattr(m, 'bias') and m.bias is not None:
    print('Zero initing bias')
    m.bias.data.zero_()

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_size, hidden_qty, nonlin_func = nn.ReLU):
      super(MLP, self).__init__()

      self.in_features = in_features
      self.out_features = out_features
      self.hidden_size = hidden_size
      self.hidden_qty = hidden_qty

      assert(self.hidden_qty >= 0)

      layers = []

      if hidden_qty == 0:

        layers = [ nn.Linear(self.in_features, self.out_features) ]

      else:

        layers.append(nn.Linear(self.in_features, self.hidden_size))
        for i in range(hidden_qty):
          layers.append(nonlin_func())
          layers.append(nn.Linear(self.hidden_size, self.hidden_size))

        layers.append(nonlin_func())
        layers.append(nn.Linear(self.hidden_size, self.out_features))

      self.mlp = nn.Sequential(*layers)

      print('MLP:', self.mlp)


    def forward(self, x):
      return self.mlp(x)

def set_freeze(module, freeze):
    
    if freeze:
        print('Freezing model!')
        requires_grad = False
    else:
        print('Unfreezing model!')
        requires_grad = True

    for one_module in module.modules():
        for param in one_module.parameters():
            param.requires_grad = requires_grad 

class ResNet(nn.Module):

    def set_freeze(self, freeze_last_block):
        set_freeze(self, True)

        if not freeze_last_block:
            set_freeze(self.avgpool, False)
            set_freeze(self.layer4, False)

        set_freeze(self.fc_custom, False)

    def __init__(self, block, layers, nb_classes, channel, freeze_last_block=False, hidden_size = 64, hidden_qty=0):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1_custom = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3,   
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)


        if USE_MLP:
            self.fc_custom = MLP(512 * block.expansion, nb_classes, 
                                 hidden_size=hidden_size, hidden_qty=hidden_qty)
        else:
            self.fc_custom = nn.Linear(512 * block.expansion, nb_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.set_freeze(freeze_last_block)

        print(self.fc_custom)


    def reset_nb_classes(self, nb_classes, avg_layers=False):
        if USE_MLP:
          m = self.fc_custom
          self.fc_custom = MLP(m.in_features, nb_classes, hidden_size=m.hidden_qty, hidden_qty=m.hidden_qty)

        else:
          self.fc_custom = nn.Linear(self.fc_custom.in_features, nb_classes)
          if avg_layers:
              if nb_classes == 2: 
                  fc_old = self.fc_custom
                  print('Reusing previously created projection matrix!')

                  w_old = fc_old.weight.data
                  w_new = self.fc_custom.weight.data

                  w_old_nr = w_old.size()[0]
                  w_new[1] = w_old[0]
                  w_new[0] = torch.sum(w_old[1:w_old_nr], dim=0)/float(w_old_nr - 1)

        print(self.fc_custom)
        

    def get_nb_classes(self):
        return self.fc_custom.out_features

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_no_last(self, x):

        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        x = self.forward_no_last(x)
        out = self.fc_custom(x)
        return out


def resnet18(pretrained, nb_classes, channel, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], nb_classes=nb_classes, channel=channel, **kwargs)
    if pretrained:
       pretrain_dict = model_zoo.load_url(model_urls['resnet18'])                  # modify pretrain code
       model_dict = model.state_dict()
       model_dict=weight_transform(model_dict, pretrain_dict, channel)
       model.load_state_dict(model_dict)

    return model


def resnet34(pretrained, nb_classes, channel, **kwargs):

    model = ResNet(BasicBlock, [3, 4, 6, 3], nb_classes=nb_classes, channel=channel, **kwargs)
    if pretrained:
       pretrain_dict = model_zoo.load_url(model_urls['resnet34'])                  # modify pretrain code
       model_dict = model.state_dict()
       model_dict=weight_transform(model_dict, pretrain_dict, channel)
       model.load_state_dict(model_dict)
    return model


def resnet50(pretrained, nb_classes, channel, **kwargs):

    model = ResNet(Bottleneck, [3, 4, 6, 3], nb_classes=nb_classes, channel=channel, **kwargs)
    if pretrained:
       pretrain_dict = model_zoo.load_url(model_urls['resnet50'])                  # modify pretrain code
       model_dict = model.state_dict()
       model_dict=weight_transform(model_dict, pretrain_dict, channel)
       model.load_state_dict(model_dict)
    return model


def resnet101(pretrained, nb_classes, channel, **kwargs):

    model = ResNet(Bottleneck, [3, 4, 23, 3],nb_classes=nb_classes, channel=channel, **kwargs)
    if pretrained:
       pretrain_dict = model_zoo.load_url(model_urls['resnet101'])                  # modify pretrain code
       model_dict = model.state_dict()
       model_dict=weight_transform(model_dict, pretrain_dict, channel)
       model.load_state_dict(model_dict)

    return model


def resnet152(pretrained, **kwargs):

    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

def cross_modality_pretrain(conv1_weight, channel):
    # transform the original 3 channel weight to "channel" channel
    S=0
    for i in range(3):
        S += conv1_weight[:,i,:,:]
    avg = S/3.
    new_conv1_weight = torch.FloatTensor(64,channel,7,7)
    #print type(avg),type(new_conv1_weight)
    for i in range(channel):
        new_conv1_weight[:,i,:,:] = avg.data
    return new_conv1_weight

def weight_transform(model_dict, pretrain_dict, channel):
    weight_dict  = {k:v for k, v in pretrain_dict.items() if k in model_dict}
    #print pretrain_dict.keys()
    w3 = pretrain_dict['conv1.weight']
    #print type(w3)
    if channel == 3:
        wt = w3
    else:
        wt = cross_modality_pretrain(w3,channel)

    weight_dict['conv1_custom.weight'] = wt
    model_dict.update(weight_dict)
    return model_dict

#Test network
if __name__ == '__main__':
    model = resnet101(pretrained= True, nb_classes=10, channel=10)
    print(model)
     
