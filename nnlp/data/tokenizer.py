

import sys
from allennlp.common import Registrable

class Tokenizer(Registrable):
    def __call__(self, text):
        pass

@Tokenizer.register('space_tokenizer')
class SpaceTokenizer(Tokenizer):
    def __init__(self, lowercase:bool=False):
        self.lowercase = lowercase

    def __call__(self, text:str):
        if self.lowercase:
            text = text.lower()
        return text.split()

@Tokenizer.register('sentencepiece_tokenizer')
class SPTokenizer(Tokenizer):
    def __init__(self, spm_path:str, lowercase:bool=False):
        import sentencepiece
        self.lowercase = lowercase

        self.spm = sentencepiece.SentencePieceProcessor()
        self.spm.Load(spm_path)

    def __call__(self, text:str):
        if self.lowercase:
            text = text.lower()
        tokens = []
        for t in self.spm.EncodeAsPieces(text):
            s= t.strip()
            if s:
                tokens.append(s)
        return tokens

@Tokenizer.register('zh_jieba_tokenizer')
class JieBaTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = False):
        import jieba
        self.lowercase = lowercase
        self.fn = lambda text: jieba.cut(text, cut_all=False)

    def __call__(self, text):
        seg_list = self.fn(text)
        seg_list = ' '.join(seg_list).split()
        return seg_list

if __name__ == '__main__':
    import sys
    from allennlp.common import Params

    p = Params(params={'type':'zh_jieba_tokenizer','lowercase':False})
    toker = Tokenizer.from_params(params=p)
    for line in sys.stdin:
        print (' '.join(toker(line.strip())))


