
import sys, random

import torch
import torch.distributed as py_dist
from torch.utils.data import IterableDataset

from allennlp.common import Registrable

import nnlp

import utils.simplelogger as simplelogger
class DataSet(Registrable):
    def __init__(self,):
        pass
@DataSet.register('iter_text_dataset')
class IterTextDataSet(DataSet, IterableDataset):
    '''
    `inputs` can be a `str` for file path, or an iterator
    '''
    def __init__(self, 
            inputs: str, 
            str2tensor_fn: nnlp.Text2Tensor =None,
            read_fn: nnlp.Reader = None,
            buff_size:int = 0,
            shuf_in_buff:bool = False,
            set_real_len:bool = False
    ):
        super(IterTextDataSet).__init__()

        self.buff_size = buff_size
        self.shuf_in_buff = shuf_in_buff
        self.inputs = inputs
        self.read_fn = read_fn
        self.str2tensor_fn = str2tensor_fn

        self.len = -1

        self.logger = simplelogger.Logger(sys.stderr, simplelogger.INFO)

        if set_real_len:
            self.len = 0
            for _ in self.gen():
                self.len += 1
            self.logger.info(f'data source: {inputs} size: {self.len}')
    def __len__(self,):
        if self.len < 0:
            raise TypeError("IterableDataset has no defined len() .")
        return self.len()

    def gen(self,):
        rank, wsize = 0, 1
        ddp_info = 'NO DDP'
        if py_dist.is_initialized():
            backend = py_dist.get_backend()
            rank = py_dist.get_rank()
            wsize = py_dist.get_world_size()
            ddp_info = f'DDP: {backend} > {rank}/{wsize}'
        assert wsize >=1 and rank < wsize
        self.logger.info(ddp_info)

        if isinstance(self.inputs, str):
            fin = open(self.inputs, encoding='utf-8')
        else:
            fin = self.inputs
        
        use_buff = self.buff_size > 1 
        buff_raw_data = []
        def get_insts(raw_data, shuf = self.shuf_in_buff):
            if shuf:
                random.shuffle(raw_data)
            for t in raw_data:
                inst = self.str2tensor_fn(t)
                if inst is not None:
                    yield inst
        for idx, s in enumerate(self.read_fn(fin)):
            #filter other worker's instances.
            if wsize > 1 and idx % wsize != rank:
                continue

            if use_buff:
                if len(buff_raw_data) >= self.buff_size:
                    for x in get_insts(buff_raw_data):
                        yield x
                    buff_raw_data = []
                else:
                    buff_raw_data.append(s)
            else:
                x = self.str2tensor_fn(s)
                if x :
                    yield x

        for x in get_insts(buff_raw_data):
            yield x
        
        fin.close()

    def __iter__(self,):
        return iter(self.gen())
    



