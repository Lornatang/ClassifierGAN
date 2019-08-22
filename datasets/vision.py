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

import os

import torch
import torch.utils.data as data


class VisionDataset(data.Dataset):
  _repr_indent = 4

  def __init__(self, root, transforms=None, transform=None, target_transform=None):
    if isinstance(root, torch._six.string_classes):
      root = os.path.expanduser(root)
    self.root = root

    has_transforms = transforms is not None
    has_separate_transform = transform is not None or target_transform is not None
    if has_transforms and has_separate_transform:
      raise ValueError("Only transforms or transform/target_transform can "
                       "be passed as argument")

    # for backwards-compatibility
    self.transform = transform
    self.target_transform = target_transform

    if has_separate_transform:
      transforms = StandardTransform(transform, target_transform)
    self.transforms = transforms

  def __getitem__(self, index):
    raise NotImplementedError

  def __len__(self):
    raise NotImplementedError

  def __repr__(self):
    head = "Dataset " + self.__class__.__name__
    body = ["Number of datapoints: {}".format(self.__len__())]
    if self.root is not None:
      body.append("Root location: {}".format(self.root))
    body += self.extra_repr().splitlines()
    if hasattr(self, "transforms") and self.transforms is not None:
      body += [repr(self.transforms)]
    lines = [head] + [" " * self._repr_indent + line for line in body]
    return '\n'.join(lines)

  def _format_transform_repr(self, transform, head):
    lines = transform.__repr__().splitlines()
    return (["{}{}".format(head, lines[0])] +
            ["{}{}".format(" " * len(head), line) for line in lines[1:]])

  def extra_repr(self):
    return ""


def _format_transform_repr(transform, head):
  lines = transform.__repr__().splitlines()
  return (["{}{}".format(head, lines[0])] +
          ["{}{}".format(" " * len(head), line) for line in lines[1:]])


class StandardTransform(object):
  def __init__(self, transform=None, target_transform=None):
    self.transform = transform
    self.target_transform = target_transform

  def __call__(self, input, target):
    if self.transform is not None:
      input = self.transform(input)
    if self.target_transform is not None:
      target = self.target_transform(target)
    return input, target

  def __repr__(self):
    body = [self.__class__.__name__]
    if self.transform is not None:
      body += _format_transform_repr(self.transform,
                                     "Transform: ")
    if self.target_transform is not None:
      body += _format_transform_repr(self.target_transform,
                                     "Target transform: ")

    return '\n'.join(body)
