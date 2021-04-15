
from allennlp.common import Registrable

class Writer(Registrable):
    def __call__(self, kv_dict_list):
        pass 

@Writer.register('text_line_writer')
class LocalTextLineWriter(Writer):
    def __init__(self, path: str, keys: str= "", sep:str = '\t'):
        self.path = path
        self.keys = keys.split()
        self.sep = spe 
    
    def __call__(self, kv_dict_list):
        with open(self.path, encoding='utf-8', mode='w') as fout:
            for d in kv_dict_list:
                res = []
                for k in self.keys:
                    assert k in d, f'key: {k} not in output dict'
                    res.append(d[k])
                print (self.sep.join(res), file = fout)
                