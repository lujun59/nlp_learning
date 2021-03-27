import sys, random

import torch
from torch.utils.data import IterableDataset

from allennlp.common import Registrable, Params

import nnlp
import utils.simplelogger as simplelogger


class DataSet(Registrable):
    def __init__(self, ):
        pass


@DataSet.register('iter_text_dataset')
class IterTextDataSet(DataSet, IterableDataset):
    def __init__(self,
                 reader: nnlp.Reader = None,
                 str2tensor_fn: nnlp.Text2Tensor = None,
                 buff_size: int = 0,
                 shuf_in_buff: bool = False,
                 set_real_len: bool = False):
        super(IterTextDataSet).__init__()

        self.buff_size = buff_size
        self.shuf_in_buff = shuf_in_buff

        self.reader = reader
        self.str2tensor_fn = str2tensor_fn

        self.len = -1

        self.logger = simplelogger.Logger(sys.stderr, simplelogger.INFO)

        if set_real_len:
            self.len = 0
            for _ in self.gen():
                self.len += 1
            self.logger.info(f'data size: {self.len}')

    def __len__(self, ):
        if self.len < 0:
            raise TypeError("IterableDataset has no defined len() .")
        return self.len()

    def gen(self, ):

        use_buff = self.buff_size > 1
        buff_raw_data = []

        def get_insts(raw_data, shuf=self.shuf_in_buff):
            if shuf:
                random.shuffle(raw_data)
            for t in raw_data:
                inst = self.str2tensor_fn(t)
                if inst:
                    yield inst

        for s in self.reader():
            if use_buff:
                if len(buff_raw_data) >= self.buff_size:
                    for x in get_insts(buff_raw_data):
                        yield x
                    buff_raw_data = []
                else:
                    buff_raw_data.append(s)
            else:
                x = self.str2tensor_fn(s)
                if x:
                    yield x

        for x in get_insts(buff_raw_data):
            yield x

    def __iter__(self, ):
        return iter(self.gen())
