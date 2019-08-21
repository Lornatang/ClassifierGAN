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

import torch.backends.cudnn as cudnn
import torch.utils.data.dataloader
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

from nets.lenet import lenet
from nets.custom_alexnet import customalexnet

parser = argparse.ArgumentParser(description='PyTorch MNIST Classifier')
parser.add_argument('--dataroot', type=str, default="~/pytorch_datasets", help="download train dataset path.")
parser.add_argument('--datasets', type=str, default="mnist", help="mnist datasets or fashion-mnist datasets.")
parser.add_argument('--phase', type=str, default='train', help="train or eval?")
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

# Load model
if opt.datasets == "mnist":
  net = lenet(nc=1, num_classes=10)
  if opt.model_path != "":
    net.load_state_dict(torch.load(opt.model_path, map_location=lambda storage, loc: storage))
  net.to(device)
elif opt.datasets == "fmnist":
  net = customalexnet(nc=1, num_classes=10)
  if opt.model_path != "":
    net.load_state_dict(torch.load(opt.model_path, map_location=lambda storage, loc: storage))
  net.to(device)
  print(net)

# Loss function
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)


def train():
  # Load model
  for epoch in range(50):
    for i, data in enumerate(train_dataloader):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      if i % 5 == 0:
        print(f"Train Epoch: {epoch} [{i * 128}/{len(train_dataloader.dataset)} "
              f"({100. * i / len(train_dataloader):.2f}%)] "
              f"Loss: {loss.item():.6f}", end="\r")
    torch.save(net.state_dict(), f"./checkpoints/{opt.datasets}_epoch_{epoch + 1}.pth")


def test(model):
  # Load model
  correct = 0.
  total = 0.
  with torch.no_grad():
    for i, data in enumerate(test_dataloader):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device)

      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

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
      print(f"Accuracy of {classes[i]:10s} : {100 * class_correct[i] / class_total[i]:.2f}%")
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
      print(f"Accuracy of {classes[i]:10s} : {100 * class_correct[i] / class_total[i]:.2f}%")


if __name__ == '__main__':

  if opt.phase == "train":
    train()
  elif opt.phase == "eval":
    if opt.model_path != "":
      print("Loading model...\n")
      net.load_state_dict(torch.load(opt.model_path, map_location=lambda storage, loc: storage))
      print("Loading model successful!")
      test(net)
      visual(net)
    else:
      print("WARNING: You want use eval pattern, so you should add --model_path MODEL_PATH")
  else:
    print(opt)
