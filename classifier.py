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

import torch

from nets.lenet import lenet
from nets.resnet_test import resnet18
from utils.cls import classifier

parser = argparse.ArgumentParser(description='PyTorch Classifier Utils!')
parser.add_argument('--datasets', type=str, default="mnist", help="Datasets name.")
parser.add_argument('--dataroot', type=str, default="test_imgs", help="Data folders to categorize.")
parser.add_argument('--img_size', type=int, default=28, help="Data folders to categorize.")
parser.add_argument('--channels', type=int, default=1, help="Number of channels in the image")
parser.add_argument('--classes_names', type=int, help="Number of channels in the image")
opt = parser.parse_args()
print(opt)

# set driver
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load model
if opt.datasets == "mnist":
  model = lenet().to(device)
elif opt.datasets == "fmnist":
  model = resnet18().to(device)

# prediction label names
if opt.datasets == "mnist":
  classes_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
elif opt.datasets == "fmnist":
  classes_names = ["T-shirt/top",
                   "Trouser",
                   "Pullover",
                   "Dress",
                   "Coat",
                   "Sandal",
                   "Skirt",
                   "Sneaker",
                   "Bag",
                   "Ankle_boot"]


def run():
  classifier(model=model,
             model_path=opt.model_path,
             datasets=opt.datasets,
             dataroot=opt.dataroot,
             img_size=opt.img_size,
             channels=opt.channels,
             classes_names=classes_names,
             device=device)


if __name__ == '__main__':
  if opt.model_path != "":
    print("Starting running......")
    run()
    print("Done..........")
  else:
    print(parser.print_help())
