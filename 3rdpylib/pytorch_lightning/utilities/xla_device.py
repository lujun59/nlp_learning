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
import functools
import os
import queue as q
import traceback
from multiprocessing import Process, Queue

from pytorch_lightning.utilities.imports import _XLA_AVAILABLE

if _XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm

#: define waiting time got checking TPU available in sec
TPU_CHECK_TIMEOUT = 25


def inner_f(queue, func, *args, **kwargs):  # pragma: no cover
    try:
        queue.put(func(*args, **kwargs))
    # todo: specify the possible exception
    except Exception:
        traceback.print_exc()
        queue.put(None)


def pl_multi_process(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        queue = Queue()
        proc = Process(target=inner_f, args=(queue, func, *args), kwargs=kwargs)
        proc.start()
        proc.join(TPU_CHECK_TIMEOUT)
        try:
            return queue.get_nowait()
        except q.Empty:
            traceback.print_exc()
            return False

    return wrapper


class XLADeviceUtils:
    """Used to detect the type of XLA device"""

    _TPU_AVAILABLE = False

    @staticmethod
    @pl_multi_process
    def _is_device_tpu() -> bool:
        """
        Check if TPU devices are available

        Return:
            A boolean value indicating if TPU devices are available
        """

        return len(xm.get_xla_supported_devices("TPU")) > 0

    @staticmethod
    def xla_available() -> bool:
        """
        Check if XLA library is installed

        Return:
            A boolean value indicating if a XLA is installed
        """
        return _XLA_AVAILABLE

    @staticmethod
    def tpu_device_exists() -> bool:
        """
        Runs XLA device check within a separate process

        Return:
            A boolean value indicating if a TPU device exists on the system
        """
        if os.getenv("PL_TPU_AVAILABLE", '0') == "1":
            XLADeviceUtils._TPU_AVAILABLE = True

        if XLADeviceUtils.xla_available() and not XLADeviceUtils._TPU_AVAILABLE:

            XLADeviceUtils._TPU_AVAILABLE = XLADeviceUtils._is_device_tpu()

            if XLADeviceUtils._TPU_AVAILABLE:
                os.environ["PL_TPU_AVAILABLE"] = '1'

        return XLADeviceUtils._TPU_AVAILABLE
