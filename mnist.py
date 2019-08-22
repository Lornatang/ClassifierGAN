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

import argparse
import os
import random
import time

import torch.backends.cudnn as cudnn
import torch.utils.data.dataloader
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

from nets.alexnet_test import alexnet
from nets.lenet import lenet
from utils.eval import accuracy
from utils.misc import AverageMeter

parser = argparse.ArgumentParser(description='PyTorch MNIST Classifier')
parser.add_argument('--dataroot', type=str, default="~/pytorch_datasets", help="download train dataset path.")
parser.add_argument('--datasets', type=str, default="mnist", help="mnist datasets or fashion-mnist datasets.")
parser.add_argument('--lr', type=float, default=0.1, help="starting lr, every 30 epoch decay 10.")
parser.add_argument('--momentum', type=float, default=0.9, help="The ratio of accelerating convergence.")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="Mainly to prevent overfitting.")
parser.add_argument('--phase', type=str, default='eval', help="train or eval?")
parser.add_argument('--model_path', type=str, default="", help="load model path.")
opt = parser.parse_args()

try:
  os.makedirs("./checkpoints")
except OSError:
  pass

manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

# setup gpu driver
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if opt.datasets == "mnist":
  # load train datasets
  train_dataset = dset.MNIST(root=opt.dataroot,
                             download=True,
                             train=True,
                             transform=torchvision.transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize([0.5], [0.5]),
                             ]))

  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                                 shuffle=True, num_workers=8)
  # load test datasets
  test_dataset = dset.MNIST(root=opt.dataroot,
                            download=True,
                            train=False,
                            transform=torchvision.transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize([0.5], [0.5]),
                            ]))

  test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                                shuffle=False, num_workers=8)
elif opt.datasets == "fmnist":
  # load train datasets
  train_dataset = dset.FashionMNIST(root=opt.dataroot,
                                    download=True,
                                    train=True,
                                    transform=torchvision.transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5], [0.5]),
                                    ]))

  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                                 shuffle=True, num_workers=8)
  # load test datasets
  test_dataset = dset.FashionMNIST(root=opt.dataroot,
                                   download=True,
                                   train=False,
                                   transform=torchvision.transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5]),
                                   ]))

  test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                                shuffle=False, num_workers=8)
else:
  print(opt)

# Load model
if opt.datasets == "mnist":
  model = lenet(nc=1, num_classes=10)
elif opt.datasets == "fmnist":
  model = alexnet(nc=1, num_classes=10)
else:
  model = ""
  print(opt)

if opt.model_path != "":
  model.load_state_dict(torch.load(opt.model_path, map_location=lambda storage, loc: storage))

model.to(device)
print(model)

# Loss function
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)


def train():
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to train mode
  model.train()

  end = time.time()
  for epoch in range(50):
    for i, data in enumerate(train_dataloader):

      # measure data loading time 计算加载数据的时间
      data_time.update(time.time() - end)

      # get the inputs; data is a list of [inputs, labels]
      inputs, targets = data
      inputs = inputs.to(device)
      targets = targets.to(device)

      # compute output
      output = model(inputs)
      loss = criterion(output, targets)

      # measure accuracy and record loss
      prec1, prec5 = accuracy(output, targets, topk=(1, 5))
      losses.update(loss.item(), inputs.size(0))
      top1.update(prec1[0], inputs.size(0))
      top5.update(prec5[0], inputs.size(0))

      # compute gradients in a backward pass
      optimizer.zero_grad()
      loss.backward()

      # Call step of optimizer to update model params
      optimizer.step()

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % 5 == 0:
        print(f"Epoch: [{epoch}] [{i}/{len(train_dataloader)}\t"
              f"Time: {data_time.val:.3f} ({data_time.avg:.3f})\t"
              f"Loss: {loss.item():.6f}\t"
              f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
              f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})")
    torch.save(model.state_dict(), f"./checkpoints/{opt.datasets}_epoch_{epoch + 1}.pth")


def test(model):
  # switch to evaluate mode
  model.eval()
  # init value
  total = 0.
  correct = 0.
  with torch.no_grad():
    for i, data in enumerate(test_dataloader):
      # get the inputs; data is a list of [inputs, labels]
      inputs, targets = data
      inputs = inputs.to(device)
      targets = targets.to(device)

      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      total += targets.size(0)
      correct += (predicted == targets).sum().item()

  print(f"\nAccuracy of the network on the 10000 test images: {100 * correct / total:.2f}%\n")


def visual(model):
  class_correct = list(0. for _ in range(10))
  class_total = list(0. for _ in range(10))

  with torch.no_grad():
    for data in test_dataloader:
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device)

      outputs = model(inputs)
      _, predicted = torch.max(outputs, 1)
      c = (predicted == labels).squeeze()

      for i in range(4):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1

  if opt.datasets == "mnist":
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for i in range(10):
      print(f"Accuracy of {classes[i]:5s} : {100 * class_correct[i] / class_total[i]:.2f}%")
  elif opt.datasets == "fmnist":
    classes = ["T-shirt/top",
               "Trouser",
               "Pullover",
               "Dress",
               "Coat",
               "Sandal",
               "Skirt",
               "Sneaker",
               "Bag",
               "Ankle_boot"]
    for i in range(10):
      print(f"Accuracy of {classes[i]:20s} : {100 * class_correct[i] / class_total[i]:.2f}%")
  else:
    pass


if __name__ == '__main__':

  if opt.phase == "train":
    train()
  elif opt.phase == "eval":
    if opt.model_path != "":
      print("Loading model...\n")
      model.load_state_dict(torch.load(opt.model_path, map_location=lambda storage, loc: storage))
      print("Loading model successful!")
      test(model)
      visual(model)
    else:
      print("WARNING: You want use eval pattern, so you should add --model_path MODEL_PATH")
  else:
    print(opt)
