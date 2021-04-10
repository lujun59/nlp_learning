import torch.utils.data

from allennlp.common import Registrable, Params

import nnlp


class DataLoader(Registrable):
    def __init__(self):
        pass


@DataLoader.register('dataloader')
class LableSentDataLoader(torch.utils.data.DataLoader, DataLoader):
    def __init__(self,
                 dataset: nnlp.DataSet,
                 collate_fn: nnlp.MakeBatch,
                 batch_size: int = 1,
                 num_workers: int = 1):
        super().__init__(dataset,
                         batch_size=batch_size,
                         collate_fn=collate_fn,
                         num_workers=num_workers)
