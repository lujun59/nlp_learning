from allennlp.common import Registrable
from allennlp.common import Params


class Vocab(Registrable):
    def __init__(self, path: str = None):

        self.token2idx = {}
        self.idx2token = []

        special_tokens = '<BOS> <EOS> <MSK> <UNK> <PAD> <SEP>'.split()
        for t in special_tokens:
            self.add_token(t)

        self.BOS = self.token2idx['<BOS>']
        self.EOS = self.token2idx['<EOS>']
        self.MSK = self.token2idx['<MSK>']
        self.UNK = self.token2idx['<UNK>']
        self.PAD = self.token2idx['<PAD>']
        self.SEP = self.token2idx['<SEP>']

        self.load(path)

    def add_token(self, t):
        assert t, f'token: <{t}> is not valid! '
        idx = self.token2idx.get(t, -1)
        if idx >= 0:
            return idx
        idx = len(self.idx2token)
        self.idx2token.append(t)
        self.token2idx[t] = idx
        return idx

    def __len__(self, ):
        assert len(self.token2idx) == len(self.idx2token)
        return len(self.idx2token)

    def get_token_id(self, t):
        return self.token2idx.get(t, self.UNK)

    def get_token_by_id(self, idx: int):
        assert 0 <= idx < len(self.idx2token)
        return self.idx2token[idx]

    def dump(self, ):
        return '\n'.join(self.idx2token)

    def dump2file(self, path: str):
        with open(path, 'w') as fout:
            fout.write(self.dump())

    def load(self, path: str):
        with open(path, encoding='utf-8') as fin:
            tokens = fin.read().split('\n')
            for t in tokens:
                if t:
                    self.add_token(t)


@Vocab.register('vocab')
class Vocab_inst(Vocab):
    def __init__(self, path: str = None):
        super().__init__(path)


def build_vocab(path):
    p = Params(params={'v': {'type': 'vocab', 'path': path}})
    return Vocab.from_params(p.pop('v'))
