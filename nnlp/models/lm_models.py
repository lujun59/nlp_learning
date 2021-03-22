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
                 n_head: int = 4,
                 n_position: int = 128,
                 n_ctx: int = 128):
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
        attention_mask = input_tensor != self.vocab.PAD

        targets = input_tensor.clone()
        targets = targets.masked_fill(~attention_mask, -100)

        output = self.gpt2_model(input_ids=input_tensor,
                                 attention_mask=attention_mask,
                                 labels=targets,
                                 use_cache=False)
        loss_batch, lm_logits, hidden_states = output[:3]

        bs, seq_len = input_tensor.size()
        seq_len -= 1

        shift_logits = lm_logits[:, :-1, :].contiguous().view(-1, self.n_vocab)
        shift_targets = targets[:, 1:].contiguous().view(-1)
        loss_token = F.cross_entropy(shift_logits,
                                     shift_targets,
                                     reduction='none').view(bs, seq_len)

        mask = attention_mask[:, 1:].float()
        loss_inst = (loss_token * mask).sum(dim=-1) / mask.sum(dim=-1)

        return loss_batch, loss_inst, loss_token


@Model.register('bigpt2_lm')
class BiGPT2LM(Model):
    def __init__(self,
                 vocab: Vocab,
                 n_embd: int = 256,
                 n_layer: int = 2,
                 n_head: int = 2,
                 n_position: int = 128,
                 n_ctx: int = 128,
                 unk_hard_loss: float = -1.0):
        super(BiGPT2LM, self).__init__()

        config = transformers.GPT2Config(vocab_size=len(vocab),
                                         n_embd=n_embd,
                                         n_layer=n_layer,
                                         n_head=n_head,
                                         n_positions=n_position,
                                         n_ctx=n_ctx,
                                         output_hidden_states=True)

        self.gpt2model_fwd = transformers.GPT2LMHeadModel(config)
        self.gpt2model_rev = transformers.GPT2LMHeadModel(config)

        self.vocab = vocab
        self.unk_hard_loss = unk_hard_loss

    def forward(self, input_tensor):
        att_mask = input_tensor != self.vocab.PAD

        def proc(inputs, gpt2model):
            t = inputs.clone()
            t = t.masked_fill(~att_mask, -100)

            output = gpt2model(input_ids=inputs,
                               attention_mask=att_mask,
                               labels=t,
                               use_cache=False)
            loss, lm_logits, hidden_states = output[:3]
            return lm_logits, hidden_states

        lm_logits_fwd, hidden_states_fwd = proc(input_tensor,
                                                self.gpt2model_fwd)

        inputs_rev = torch.flip(input_tensor, (1, ))
        lm_logits_rev, hidden_states_rev = proc(inputs_rev, self.gpt2model_rev)
        lm_logits_rev = torch.flip(lm_logits_rev, (1, ))

        batch_size, seq_len = input_tensor.size()
        seq_len -= 2

        shift_logits_fwd = lm_logits_fwd[:, :-2, :].contiguous().view(
            batch_size * seq_len,
            lm_logits_fwd.size()[-1])
        shift_logits_rev = lm_logits_rev[:, 2:, :].contiguous().view(
            batch_size * seq_len,
            lm_logits_rev.size()[-1])
        shift_logits_bi = torch.cat((shift_logits_fwd, shift_logits_rev), -1)
        shift_labels = input_tensor[:, 1:-1].contiguous().view(
            batch_size * seq_len, )

        loss_token = F.cross_entropy(shift_logits_bi,
                                     shift_labels,
                                     reduction='none')
        loss_token = loss_token.view(batch_size, seq_len)

        if self.unk_hard_loss > 0.0:
            unk_mask = input_tensor[:, 1:-1] == self.vocab.UNK
            loss_token = loss_token.masked_fill(unk_mask, self.unk_hard_loss)

        mask = att_mask[:, 1:-1].float()
        loss_line = (loss_token * mask).sum(dim=-1) / mask.sum(dim=-1)
        loss_batch = (loss_token * mask).sum() / mask.sum()

        return loss_batch, loss_line, loss_token


@Model.register('bert_lm')
class BERTLMModel(Model):
    def __init__(self,
                 vocab: Vocab,
                 n_embd: int = 256,
                 n_layer: int = 4,
                 n_head: int = 4,
                 n_position: int = 128,
                 n_ctx: int = 128):
        super(BERTLMModel, self).__init__()

        config = transformers.BertConfig(vocab_size=len(vocab),
                                         hidden_size=n_embd,
                                         num_hidden_layers=n_layer,
                                         num_attention_heads=n_head,
                                         output_hidden_states=True)

        self.bert_model = transformers.BertForMaskedLM(config)
        self.vocab = vocab
        self.mlm_probability = 0.15

    def mask_tokens(self, inputs: torch.Tensor):
        device = inputs.device
        labels = inputs.clone()
        shape = labels.size()

        prob_matrix = torch.full(shape, self.mlm_probability, device=device)

        special_token_mask = labels.eq(self.vocab.BOS) | labels.eq(
            self.vocab.EOS) | labels.eq(self.vocab.PAD) | labels.eq(
                self.vocab.SEP)
        prob_matrix.masked_fill_(special_token_mask, value=0.0)
        masked_ids = torch.bernoulli(prob_matrix).bool()
        labels[~masked_ids] = -100

        # 80% of the time, we replace masked input tokens with mask token (<MSK>)
        ids_replace = torch.bernoulli(
            torch.full(labels.shape, 0.8, device=device)).bool() & masked_ids
        inputs[ids_replace] = self.vocab.MSK

        # 10% of the time, we replace masked input tokens with random word
        ids_random = torch.bernoulli(
            torch.full(labels.shape, 0.5,
                       device=device)).bool() & masked_ids & ~ids_replace
        random_word_ids = torch.randint(len(self.vocab),
                                        labels.shape,
                                        dtype=torch.long,
                                        device=device)
        inputs[ids_random] = random_word_ids[ids_random]

        return inputs, labels

    def forward(self, input_tensor):
        inputs, labels = self.mask_tokens(input_tensor)
        att_mask = input_tensor != self.vocab.PAD

        return self.bert_model(input_ids=inputs,
                               attention_mask=att_mask,
                               labels=labels)[0], None, None
