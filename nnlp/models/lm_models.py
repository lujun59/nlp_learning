import torch
import torch.nn as nn
import torch.nn.functional as F

from nnlp.models.model import Model
from nnlp.data.vocab import Vocab

import transformers


@Model.register('lstm_lm')
class LSTMModel(Model):
    def __init__(self,
                 vocab: Vocab,
                 n_wemb: int,
                 n_hidden: int,
                 n_layer: int = 1,
                 dropout: float = 0.2,
                 tie_weights: bool = False):
        super(LSTMModel, self).__init__()

        n_vocab = len(vocab)
        #print (n_vocab)
        self.embedder = nn.Embedding(n_vocab, n_wemb)
        self.rnn = nn.LSTM(input_size=n_wemb,
                           hidden_size=n_hidden,
                           num_layers=n_layer,
                           batch_first=True,
                           dropout=dropout)

        self.decoder = nn.Linear(n_hidden, n_vocab)
        self.dropout = nn.Dropout(dropout)

        if tie_weights:
            if n_hidden != n_wemb:
                raise ValueError(
                    'When using the tied flag, n_hidden must be equal to n_wemb'
                )
            self.decoder.weight = self.embedder.weight

        self.n_layer = n_layer
        self.n_hidden = n_hidden
        self.n_vocab = n_vocab
        self.vocab = vocab

    def forward(self, input_tensor):
        '''
            input_tensor: [batch_size, seq_len]
        '''
        #print(input_tensor.size())
        batch_size, seq_len = input_tensor.size()
        mask = input_tensor[:,
                            2:] != self.vocab.PAD  # [batch_size, seq_len-2 ]

        targets = input_tensor[:, 1:-1].reshape(
            (batch_size * (seq_len - 2), ))  # [batch_size *(seq_len-2) ]

        emb = self.dropout(
            self.embedder(input_tensor))  #[batch_size, seq_len, n_wemb]
        output, hidden = self.rnn(emb)
        output = output[:, :-2, :]  #[batch_size, seq_len-2, n_hidden]
        decoded = self.decoder(output)  #[batch_size, seq_len-2, n_vocab]
        decoded = decoded.view(-1, self.n_vocab)

        loss_token = F.cross_entropy(decoded, targets, reduction='none')
        loss_token = loss_token.view(batch_size, seq_len - 2)

        loss_inst = (loss_token *
                     mask.float()).sum(dim=-1) / mask.sum(dim=-1).float()
        loss_batch = (loss_token * mask.float()).sum() / mask.sum().float()

        return loss_batch, loss_inst, loss_token


@Model.register('gpt2_lm')
class GPT2Wrap(Model):
    def __init__(self,
                 vocab: Vocab,
                 n_embd: int = 256,
                 n_layer: int = 4,
                 n_head=4,
                 n_position=128,
                 n_ctx=128):
        super(GPT2Wrap, self).__init__()

        config = transformers.GPT2Config(vocab_size=len(vocab),
                                         n_embd=n_embd,
                                         n_layer=n_layer,
                                         n_head=n_head,
                                         n_positions=n_position,
                                         n_ctx=n_ctx,
                                         output_hidden_states=True)

        self.gpt2_model = transformers.GPT2LMHeadModel(config)
        self.vocab = vocab
        self.n_vocab = len(vocab)

    def forward(self, input_tensor):
        attention_mask = input_tensor != vocab.PAD

        targets = input_tensor.clone()
        targets = targets.masked_fill(~attention_mask, -100)

        output = self.gpt2_model(input_ids=input_tensor,
                                 attention_mask=attention_mask,
                                 labels=targets,
                                 use_cache=False)
        loss_batch, lm_logits, hidden_states = output[:3]

        bs, seq_len = input_tensor.size()
        seq_len -= 1

        shift_logits = lm_logits[:, :-1, :].contiguous().view(-1, n_vocab)
        shift_targets = targets[:, 1:].contiguous().view(-1)
        loss_token = F.cross_entropy(shift_logits,
                                     shift_targets,
                                     reduction='none').view(bs, seq_len)

        mask = attention_mask[:, 1:].float()
        loss_inst = (loss_token * mask).sum(dim=-1) / mask.sum(dim=-1)

        return loss_batch, loss_inst, loss_token