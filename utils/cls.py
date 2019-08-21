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


def classifier(model, dataloader, classes, device):
  """ A classification function used to classify pictures of locations.
  Args:
    model: Neural network structure loaded by classifier.
    dataloader: Data flows that need to be classified.
    classes: Multiple picture tags.
    device: Set whether to use a GPU
  Examples:
    >> train_dataset = dset.MNIST(**kwargs)
    >> train_dataloader = torch.utils.data.DataLoader(train_dataset, **kwargs)
    >> model = Model(**kwargs)
    >> classifier(model, train_dataloader)
  """
  for i in range(len(classes)):
    try:
      os.makedirs(classes[i])
    except OSError:
      pass
  with torch.no_grad():
    for i, data in enumerate(dataloader):
      # get the inputs; data is a list of [inputs, labels]
      inputs, _ = data
      inputs = inputs.to(device)

      outputs = model(inputs)
      _, predicted = torch.max(outputs, 1)
