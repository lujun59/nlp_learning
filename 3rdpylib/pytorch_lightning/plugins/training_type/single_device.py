# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Optional, Union

import torch

from pytorch_lightning.plugins.training_type.training_type_plugin import TrainingTypePlugin


class SingleDevicePlugin(TrainingTypePlugin):

    def __init__(self, device: torch.device):
        super().__init__()
        self.device: torch.device = device
        self.global_rank = 0
        self.local_rank = 0
        self.world_size = 1

    @property
    def on_tpu(self) -> bool:
        return False

    @property
    def on_gpu(self) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_available()

    def reduce(self, output: Union[Any, torch.Tensor], *args: Any, **kwargs: Any) -> Union[Any, torch.Tensor]:
        """
        Reduces output from several distributed processes to one aggregated tensor.
        As this plugin only operates with a single device, the reduction is simply the identity.

        Args:
            output: the tensor to sync and reduce
            *args: ignored
            **kwargs: ignored

        Return:
            the unmodified input as reduction is not needed for single process operation
        """
        return output

    def all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> torch.Tensor:
        """Perform a all_gather on all processes """
        return tensor

    @property
    def root_device(self) -> torch.device:
        return self.device

    def model_to_device(self) -> None:
        if self.on_gpu:
            torch.cuda.set_device(self.root_device)

        self._model.to(self.root_device)

    def connect(self, model: torch.nn.Module) -> torch.nn.Module:
        self._model = model
        self.model_to_device()
        return self.model

    @property
    def is_global_zero(self) -> bool:
        return True

    def barrier(self, *args, **kwargs) -> None:
        pass

    def broadcast(self, obj: object, src: int = 0) -> object:
        return obj
