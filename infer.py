import sys
import json
from typing import Dict

import torch
import torch.utils.data as torch_data

from allennlp.common import Registrable, Params
from allennlp.common.util import params_from_file

import utils.simplelogger as simplelogger

import nnlp


class Infer(Registrable):
    def __init__(self):
        pass

    def model_load(self, fmt='torch', in_path: str= None):
        assert fmt in ('torch', 'torchscript', 'onnx')
        if fmt == 'torch':
            assert in_path
            self.model.load_state_dict(torch.load(in_path))
        
    
@Infer.register('lm_infer')
class LMInfer(Infer):
    def __init__(self,
                 input_dataloader: nnlp.DataLoader,
                 out_writer: nnlp.Writer,
                 model: nnlp.Model,
                 model_chk_path: str,
                 model_fmt: str='torch'):
        
        self.mylogger = simplelogger.Logger(sys.stderr)

        self.device = torch.device('cuda')
        self.in_data = input_dataloader
        self.out_writer = out_writer

        self.model = model
        self.model_load(fmt=model_fmt, in_path=model_chk_path)
        self.model.to(self.device)
        self.model.eval()

    def infer(self):
        for x_batch, y_batch, meta in self.in_data:
            outputs = self.model(x_batch)
            loss_batch, loss_insts, loss_tokens = outputs[:3]
            assert len(x_batch) == len(loss_insts) == len(meta)

            for loss_one, m in zip(loss_insts, meta ):
                pass

        


if __name__ == '__main__':

    cfg_path = sys.argv[1]

    params = params_from_file(cfg_path, 'jsonnet')
    print(json.dumps(dict(params.as_dict()), indent=2), file=sys.stderr)

    lmt = Experiment.from_params(Params(params=params))
    lmt.run_train()
