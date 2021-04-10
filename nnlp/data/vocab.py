import sys
from allennlp.common import Registrable, Params
import utils.simplelogger as simplelogger

#TODO: 单例模式避免重复构建词表


class Vocab(Registrable):
    def __init__(self, path: str = None, reserved_tokens: str = ''):
        self.mylogger = simplelogger.Logger(sys.stderr)

        self.token2idx = {}
        self.idx2token = []

        special_tokens = '<BOS> <EOS> <MSK> <UNK> <PAD> <SEP> <CLS>' + reserved_tokens
        for t in special_tokens.split():
            self.add_token(t)

        self.BOS = self.token2idx['<BOS>']
        self.EOS = self.token2idx['<EOS>']
        self.MSK = self.token2idx['<MSK>']
        self.UNK = self.token2idx['<UNK>']
        self.PAD = self.token2idx['<PAD>']
        self.SEP = self.token2idx['<SEP>']
        self.SEP = self.token2idx['<CLS>']

        if path:
            self.load(path)
            self.mylogger.info(
                f'vocab built, size: {len(self)} , from file: {path}')

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
        n = len(self.idx2token)
        assert len(self.token2idx) == n
        return n

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
    def __init__(self, path: str = None, reserved_tokens: str = ''):
        super().__init__(path, reserved_tokens)


def build_vocab(path):
    p = Params(params={'v': {'type': 'vocab', 'path': path}})
    return Vocab.from_params(p.pop('v'))
