# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from dassl.modeling.backbone import BACKBONE_REGISTRY, Backbone

from .mixstyle import MixStyle2 as MixStyle

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
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
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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

class Intra_ADR(nn.Module):
    def __init__(self, outp, Norm=None, group=1, stride=2, **kwargs):
        super(Intra_ADR, self).__init__()
        self.E_space = nn.Sequential(
            nn.ConvTranspose2d(outp, outp, kernel_size=2, stride=stride, padding=0, output_padding=0, groups=1,
                            bias=True, dilation=1, padding_mode='zeros'),
            nn.InstanceNorm2d(outp),
            nn.ReLU(inplace=True),
            )
        self.mixstyle = MixStyle(p=.5, alpha=.3)
        
    def cc_kth_p(self, input, kth=0):
        kth = 10
        input = torch.topk(input, kth, dim=1)[0]  # n,k,h,w

        input = input.mean(1, keepdim=True)
        return input

    def forward(self, x):
        branch = self.E_space(x)
        branch2 = branch
        x_adr = branch
        branch_ = branch.reshape(branch.size(0), branch.size(1), branch.size(2) * branch.size(3))
        branch = F.softmax(branch_, 2)
        branch_out = self.cc_kth_p(branch)
        return branch_out, branch2, x_adr
    
class ResNet(Backbone):

    def __init__(self, block, layers, mixstyle_layers=[], mixstyle_p=0.5, mixstyle_alpha=0.3, **kwargs):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        

        self.mixstyle = None
        if mixstyle_layers:
            self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
            for layer_name in mixstyle_layers:
                assert layer_name in ['conv1', 'conv2_x', 'conv3_x', 'conv4_x', 'conv5_x']
            print('Insert MixStyle after the following layers: {}'.format(mixstyle_layers))
        self.mixstyle_layers = mixstyle_layers

        self._out_features = 512 * block.expansion
        
        self.intra_adr = Intra_ADR(self._out_features, Norm=self.mixstyle)

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     if m.bias is not None:
            #         nn.init.constant_(m.bias, 0)

    def compute_style(self, x):
        mu = x.mean(dim=[2, 3])
        sig = x.std(dim=[2, 3])
        return torch.cat([mu, sig], 1)

    def ccmp(self, input, kernel_size, stride):
        input = input.permute(0, 3, 2, 1)
        input = F.max_pool2d(input, kernel_size, stride)
        input = input.permute(0, 3, 2, 1).contiguous()
        return input
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if 'conv2_x' in self.mixstyle_layers:
            x = self.mixstyle(x)
        fm2 = x
        x = self.layer2(x)
        if 'conv3_x' in self.mixstyle_layers:
            x = self.mixstyle(x)
        fm3 = x
        x = self.layer3(x)
        if 'conv4_x' in self.mixstyle_layers:
            x = self.mixstyle(x)
        fm4 = x
        x = self.layer4(x)
        if 'conv5_x' in self.mixstyle_layers:
            x = self.mixstyle(x)
        fm5 = x
        branch_out, branch2, x_adr = self.intra_adr(x)
        x_ce = x
        # fm = x
        b2_out = self.gmp(branch2)
        b2_out = b2_out.view(b2_out.size(0), -1)
        x_adr = nn.Dropout(.0)(self.global_avgpool(x_adr))
        x_ce = nn.Dropout(.0)(self.global_avgpool(x_ce))
        x_adr = x_adr.view(x_adr.size(0), -1)
        x_ce = x_ce.view(x_ce.size(0), -1)
        
        return [x_adr, x_ce], [branch_out, b2_out], [fm2, fm3, fm4, fm5]


@BACKBONE_REGISTRY.register()
def resnet50_ms_L23(pretrained=True, **kwargs):
    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        mixstyle_layers=['conv2_x', 'conv3_x'],
        mixstyle_p=0.5,
        mixstyle_alpha=0.1
    )
    return model

