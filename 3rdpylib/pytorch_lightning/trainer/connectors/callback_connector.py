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
import os
from typing import List, Union

from pytorch_lightning.callbacks import Callback, ModelCheckpoint, ProgressBar, ProgressBarBase
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class CallbackConnector:

    def __init__(self, trainer):
        self.trainer = trainer

    def on_trainer_init(
        self,
        callbacks,
        checkpoint_callback,
        progress_bar_refresh_rate,
        process_position,
        default_root_dir,
        weights_save_path,
        resume_from_checkpoint,
        stochastic_weight_avg,
    ):
        self.trainer.resume_from_checkpoint = resume_from_checkpoint

        # init folder paths for checkpoint + weights save callbacks
        self.trainer._default_root_dir = default_root_dir or os.getcwd()
        self.trainer._weights_save_path = weights_save_path or self.trainer._default_root_dir
        self.trainer._stochastic_weight_avg = stochastic_weight_avg

        # init callbacks
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]
        self.trainer.callbacks = callbacks or []

        # configure checkpoint callback
        # pass through the required args to figure out defaults
        self.configure_checkpoint_callbacks(checkpoint_callback)

        # configure swa callback
        self._configure_swa_callbacks()

        # init progress bar
        self.trainer._progress_bar_callback = self.configure_progress_bar(progress_bar_refresh_rate, process_position)

        # push all checkpoint callbacks to the end
        # it is important that these are the last callbacks to run
        self.trainer.callbacks = self._reorder_callbacks(self.trainer.callbacks)

    def configure_checkpoint_callbacks(self, checkpoint_callback: Union[ModelCheckpoint, bool]):
        if isinstance(checkpoint_callback, ModelCheckpoint):
            # TODO: deprecated, remove this block in v1.3.0
            rank_zero_warn(
                "Passing a ModelCheckpoint instance to Trainer(checkpoint_callbacks=...)"
                " is deprecated since v1.1 and will no longer be supported in v1.3."
                " Use `callbacks` argument instead.", DeprecationWarning
            )
            self.trainer.callbacks.append(checkpoint_callback)

        if self._trainer_has_checkpoint_callbacks() and checkpoint_callback is False:
            raise MisconfigurationException(
                "Trainer was configured with checkpoint_callback=False but found ModelCheckpoint"
                " in callbacks list."
            )

        if not self._trainer_has_checkpoint_callbacks() and checkpoint_callback is True:
            self.trainer.callbacks.append(ModelCheckpoint(dirpath=None, filename=None, mode='min'))

    def _configure_swa_callbacks(self):
        if not self.trainer._stochastic_weight_avg:
            return

        from pytorch_lightning.callbacks.swa import StochasticWeightAveraging
        existing_swa = [cb for cb in self.trainer.callbacks if isinstance(cb, StochasticWeightAveraging)]
        if not existing_swa:
            self.trainer.callbacks = [StochasticWeightAveraging()] + self.trainer.callbacks

    def configure_progress_bar(self, refresh_rate=None, process_position=0):
        if os.getenv('COLAB_GPU') and refresh_rate is None:
            # smaller refresh rate on colab causes crashes, choose a higher value
            refresh_rate = 20
        refresh_rate = 1 if refresh_rate is None else refresh_rate

        progress_bars = [c for c in self.trainer.callbacks if isinstance(c, ProgressBarBase)]
        if len(progress_bars) > 1:
            raise MisconfigurationException(
                'You added multiple progress bar callbacks to the Trainer, but currently only one'
                ' progress bar is supported.'
            )
        elif len(progress_bars) == 1:
            progress_bar_callback = progress_bars[0]
        elif refresh_rate > 0:
            progress_bar_callback = ProgressBar(
                refresh_rate=refresh_rate,
                process_position=process_position,
            )
            self.trainer.callbacks.append(progress_bar_callback)
        else:
            progress_bar_callback = None

        return progress_bar_callback

    def _trainer_has_checkpoint_callbacks(self):
        return len(self.trainer.checkpoint_callbacks) > 0

    def attach_model_logging_functions(self, model):
        for callback in self.trainer.callbacks:
            callback.log = model.log
            callback.log_dict = model.log_dict

    @staticmethod
    def _attach_model_callbacks(model: LightningModule, trainer) -> None:
        """
        Attaches the callbacks defined in the model.
        If a callback returned by the model's configure_callback method has the same type as one or several
        callbacks already present in the trainer callbacks list, it will replace them.
        In addition, all :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` callbacks
        will be pushed to the end of the list, ensuring they run last.

        Args:
            model: A model which may or may not define new callbacks in
                :meth:`~pytorch_lightning.core.lightning.LightningModule.configure_callbacks`.
            trainer: The trainer on which the callbacks get attached/merged.
        """
        model_callbacks = model.configure_callbacks()
        if not model_callbacks:
            return
        model_callback_types = set(type(c) for c in model_callbacks)
        trainer_callback_types = set(type(c) for c in trainer.callbacks)
        override_types = model_callback_types.intersection(trainer_callback_types)
        if override_types:
            rank_zero_info(
                "The following callbacks returned in `LightningModule.configure_callbacks` will override"
                " existing callbacks passed to Trainer:"
                f" {', '.join(sorted(t.__name__ for t in override_types))}"
            )
        # remove all callbacks with a type that occurs in model callbacks
        all_callbacks = [c for c in trainer.callbacks if type(c) not in override_types]
        all_callbacks.extend(model_callbacks)
        all_callbacks = CallbackConnector._reorder_callbacks(all_callbacks)
        # TODO: connectors refactor: move callbacks list to connector and do not write Trainer state
        trainer.callbacks = all_callbacks

    @staticmethod
    def _reorder_callbacks(callbacks: List[Callback]) -> List[Callback]:
        """
        Moves all ModelCheckpoint callbacks to the end of the list. The sequential order within the group of
        checkpoint callbacks is preserved, as well as the order of all other callbacks.

        Args:
            callbacks: A list of callbacks.

        Return:
            A new list in which the last elements are ModelCheckpoints if there were any present in the
            input.
        """
        checkpoints = [c for c in callbacks if isinstance(c, ModelCheckpoint)]
        not_checkpoints = [c for c in callbacks if not isinstance(c, ModelCheckpoint)]
        return not_checkpoints + checkpoints
