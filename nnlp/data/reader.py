

from allennlp.common import Registrable
from allennlp.common import Params


class Reader(Registrable):
    def __call__(self, fin):
        for line in fin:
            yield line.strip()

@Reader.register('text_line_reader')
class TextLineReader(Reader):
    def __init__(self, strip = True, skip_empty_line = True):
        self.strip = strip
        self.skip_empty_line = skip_empty_line 
    def __call__(self, fin):
        for line in fin:
            line = line.rstrip('\n\r')
            if self.strip:
                line = line.strip()
            if self.skip_empty_line and not line:
                continue
            yield line

            
