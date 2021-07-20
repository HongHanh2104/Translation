import torch
from torch import nn
import torch.nn.functional as F


class TokenCrossEntropyLoss(nn.Module):
    def __init__(self, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.base_cross_entropy = nn.CrossEntropyLoss(reduction='sum',
                                                      ignore_index=pad_idx)

    def forward(self, out, trg):
        b, seq_len, vocab_size = out.shape

        ntokens = (trg != self.pad_idx).sum().item()

        out = out.reshape(b * seq_len, vocab_size)
        trg = trg.reshape(b * seq_len)

        if False:
            loss = self.base_cross_entropy(out, trg)
        else:
            eps = 0.1
            n_class = out.size(1)

            one_hot = torch.zeros_like(out).scatter(1, trg.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(out, dim=1)

            non_pad_mask = trg.ne(self.pad_idx)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()

        return loss / ntokens, ntokens
