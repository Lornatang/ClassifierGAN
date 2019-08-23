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

from .mnist import EMNIST
from .mnist import FashionMNIST
from .mnist import KMNIST
from .mnist import MNIST

from .cifar import CIFAR10
from .cifar import CIFAR100

__all__ = ('EMNIST', 'FashionMNIST', 'KMNIST', 'MNIST', 'CIFAR10', 'CIFAR100')
