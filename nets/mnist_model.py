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

"""AlexNet model for classifier MNIST and Fashion-MNIST"""

import torch
import torch.nn as nn

__all__ = ['AlexNet', 'alexnet']


class AlexNet(nn.Module):
  """ On the basis of the original improvement"""

  def __init__(self, num_classes=10, ngpu=1):
    super(AlexNet, self).__init__()

    self.ngpu = ngpu

    self.features = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 3 * 3, 1024),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(1024, 512),
      nn.ReLU(inplace=True),
      nn.Linear(512, num_classes),
    )

  def forward(self, x):
    if x.is_cuda and self.ngpu > 1:
      x = nn.parallel.data_parallel(self.features, x, range(self.ngpu))
      x = torch.flatten(x, 1)
      x = nn.parallel.data_parallel(self.classifier, x, range(self.ngpu))
    else:
      x = self.features(x)
      x = torch.flatten(x, 1)
      x = self.classifier(x)
    return x


def alexnet(**kwargs):
  r"""AlexNet model architecture from the
  `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
  """
  model = AlexNet(**kwargs)
  return model
