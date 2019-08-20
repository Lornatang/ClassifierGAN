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

import random

import torch.backends.cudnn as cudnn
import torch.utils.data.dataloader
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

from nets.mnist_model import AlexNet

manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available():
  print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# load train datasets
train_dataset = dset.MNIST(root="~/pytorch_datasets",
                           download=True,
                           train=True,
                           transform=torchvision.transforms.Compose([
                             transforms.Resize(32),
                             transforms.RandomCrop(28),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.RandomGrayscale(),
                             transforms.Normalize([0.5], [0.5]),
                           ]))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                               shuffle=True, num_workers=8)
# load test datasets
test_dataset = dset.MNIST(root=DATA_ROOT,
                          download=True,
                          train=False,
                          transform=torchvision.transforms.Compose([
                            transforms.ToTensor(),
                          ]))

test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                              shuffle=False, num_workers=8)


net = AlexNet()