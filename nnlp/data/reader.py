import torch
import torch.distributed as py_dist
from allennlp.common import Registrable, Params


class Reader(Registrable):
    def __call__(self):
        pass


@Reader.register('text_line_reader')
class LocalTextLineReader(Reader):
    def __init__(self, path: str, strip=True, skip_empty_line=True):
        self.path = path
        self.strip = strip
        self.skip_empty_line = skip_empty_line

    def __call__(self):
        rank, wsize = 0, 1
        ddp_info = 'NO DDP'
        if py_dist.is_initialized():
            backend = py_dist.get_backend()
            rank = py_dist.get_rank()
            wsize = py_dist.get_world_size()
            ddp_info = f'DDP: {backend} > {rank}/{wsize}'
        assert wsize >= 1 and rank < wsize

        num_workers, worker_id = 1, 0
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            num_workers, worker_id = worker_info.num_workers, worker_info.id

        wsize = wsize * num_workers
        rank = rank * num_workers + worker_id

        with open(self.path, encoding='utf-8') as fin:
            idx = -1
            for line in fin:
                idx += 1
                line = line.rstrip('\n\r')
                if self.strip:
                    line = line.strip()
                if self.skip_empty_line and not line:
                    continue

                #skip other worker's instances.
                if wsize > 1 and idx % wsize != rank:
                    continue

                yield line
