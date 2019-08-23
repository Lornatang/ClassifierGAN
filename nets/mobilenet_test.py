# Copyright 2019 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MobileNetV2', 'mobilenet_v2']


class ConvBNReLU(nn.Sequential):
  def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
    padding = (kernel_size - 1) // 2
    super(ConvBNReLU, self).__init__(
      nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
      nn.BatchNorm2d(out_planes),
      nn.ReLU6(inplace=True)
    )


class Block(nn.Module):
  """expand + depthwise + pointwise"""

  def __init__(self, in_planes, out_planes, expansion, stride):
    super(Block, self).__init__()
    self.stride = stride

    planes = expansion * in_planes
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn3 = nn.BatchNorm2d(out_planes)

    self.shortcut = nn.Sequential()
    if stride == 1 and in_planes != out_planes:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_planes),
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    out = out + self.shortcut(x) if self.stride == 1 else out
    return out


class MobileNetV2(nn.Module):
  # (expansion, out_planes, num_blocks, stride)
  cfg = [(1, 16, 1, 1),
         (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
         (6, 32, 3, 2),
         (6, 64, 4, 2),
         (6, 96, 3, 1),
         (6, 160, 3, 2),
         (6, 320, 1, 1)]

  def __init__(self, num_classes=100):
    super(MobileNetV2, self).__init__()

    self.features = nn.Sequential(
      ConvBNReLU(3, 32),
      self._make_layers(in_planes=32),
      ConvBNReLU(320, 1280, kernel_size=1),
    )
    # NOTE: change conv1 stride 2 -> 1 for CIFAR10

    # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    # self.bn1 = nn.BatchNorm2d(32)

    # self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
    # self.bn2 = nn.BatchNorm2d(1280)
    self.classifier = nn.Linear(1280, num_classes)

  def _make_layers(self, in_planes):
    layers = []
    for expansion, out_planes, num_blocks, stride in self.cfg:
      strides = [stride] + [1] * (num_blocks - 1)
      for _stride in strides:
        layers.append(Block(in_planes, out_planes, expansion, _stride))
        in_planes = out_planes
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.features(x)
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    out = self.classifier(out)
    return out


def mobilenet_v2(**kwargs):
  """
  Constructs a MobileNetV2 architecture from
  `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

  """
  model = MobileNetV2(**kwargs)
  return model
