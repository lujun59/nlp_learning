
import torch

from allennlp.common import Registrable

from nnlp.data.vocab import Vocab
from nnlp.data.tokenizer import Tokenizer


class Text2Tensor(Registrable):
    def __call__(self, ):
        pass

class MakeBatch(Registrable):
    def __call__(self,):
        pass

@Text2Tensor.register('label_sent2tensor')
class LableSent2Tensor(Text2Tensor):
    def __init__(self, 
            vocab:Vocab,
            tokenizer:Tokenizer=None, 
            max_len:int = 80, 
            with_lable:bool=False,
            lable_type:str = 'float'):

        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = max_len
        self.with_lable = with_lable
        self.label_type = lable_type

        assert lable_type in ('str','int','float')

    def __call__(self, text):
        '''
        `text`'s format with lable=True: [tag] \t [raw text]
        `text`'s format with lable=False: [raw text]
        '''
        ps = text.strip().split('\t')
        sent = ''
        tag = '0'
        
        if self.with_lable:
            assert len(ps) == 2
            tag, sent = ps
        else:
            assert len(ps) == 1
            sent = text.strip()
        
        if not sent:
            return None
        if not self.tokenizer:
            tokens = sent.split()
        else:
            tokens = self.tokenizer(sent)
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]

        tag_tensor_type = torch.long
        if self.label_type == 'int':
            tag = int(tag)
        elif self.label_type == 'float':
            tag = float(tag)
            tag_tensor_type = torch.float
        elif self.label_type == 'str':
            tag = self.vocab.get_token_id(tag)

        meta = {'sent':sent, 'tokens': tokens, 'label':tag }
        tokens_idx = [self.vocab.BOS] + [self.vocab.get_token_id(w) for w in tokens] + [self.vocab.EOS]
        return (torch.tensor(tokens_idx, dtype=torch.long), torch.tensor(tag, dtype = tag_tensor_type), meta)


@MakeBatch.register('label_sent2batch')
class LableSent2Batch(MakeBatch):
    def __init__(self, pad_token_id:int):
        self.pad_token_id = pad_token_id
    
    def __call__(self, insts):

        max_len = 0
        token_idx_list = []
        tags = []
        meta =[]
        for t in insts:
            if not t: continue
            token_idx_list.append(t[0])
            tags.append(t[1])
            meta.append(t[2])
            if max_len < len(t[0]):
                max_len = len(t[0])
        batch_token_idx = torch.full((len(token_idx_list), max_len), self.pad_token_id, dtype=torch.long)
        for i, t in enumerate(token_idx_list):
            batch_token_idx[i, :len(t)] = t
        tags = torch.stack(tags)
        return (batch_token_idx, tags, meta)
