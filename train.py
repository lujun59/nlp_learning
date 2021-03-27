import sys
import json
from typing import Dict

import torch
import torch.utils.data as torch_data

from allennlp.common import Registrable, Params
from allennlp.common.util import params_from_file

import utils.simplelogger as simplelogger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import nnlp


class ModelTrainer(Registrable, pl.LightningModule):
    def __init__(self):
        super().__init__()
        pass


@ModelTrainer.register('lm_trainer')
class LMTrain(ModelTrainer):
    def __init__(self,
                 train_dataloader: nnlp.DataLoader,
                 valid_dataloader: nnlp.DataLoader,
                 model: nnlp.Model,
                 out_path: str,
                 early_stop: bool = False,
                 train_params: Dict = {}):
        super().__init__()

        self.mylogger = simplelogger.Logger(sys.stderr)

        #self.mylogger.info(f'vocab size: {len(vocab)}')
        #self.mylogger.info(f'train data path: {train_data_path}')
        #self.mylogger.info(f'valid data path: {valid_data_path}')

        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        self.model = model
        self.train_params = train_params

        self.callback_checkpoint = ModelCheckpoint(
            monitor="val_batch_loss",
            dirpath=f"{out_path}",
            filename='chk-{epoch:02d}-{step:03d}-{val_batch_loss:.2f}',
            save_top_k=20,
            mode='min')

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        loss_batch, loss_inst, loss_token = self.model(x)
        self.log('train_loss', loss_batch)
        self.log('global_step', self.global_step)
        return loss_batch

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        loss_batch, loss_inst, loss_token = self.model(x)
        self.log('val_batch_loss', loss_batch, prog_bar=True, sync_dist=True)
        return loss_batch

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def run_train(self, ):
        trainer = pl.Trainer(**self.train_params,
                             checkpoint_callback=self.callback_checkpoint)
        trainer.fit(self, self.train_data, self.valid_data)


if __name__ == '__main__':

    cfg_path = sys.argv[1]

    params = params_from_file(cfg_path, 'jsonnet')
    print(json.dumps(dict(params.as_dict()), indent=2), file=sys.stderr)

    lmt = ModelTrainer.from_params(Params(params=params))
    lmt.run_train()
