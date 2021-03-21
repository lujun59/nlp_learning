

import torch.nn as nn

from allennlp.common import Registrable

class Model(Registrable, nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
